import torch.nn.functional as F
from reid.loss.softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from reid.loss.triplet_loss_transreid import TripletLoss
from reid.loss.center_loss import CenterLoss
import torch
def make_loss(cfg, num_classes):    # modified by gu
    """
    根据配置文件 cfg 生成损失函数。
    
    参数:
    - cfg: 配置对象，包含模型训练时所需的所有配置信息。
    - num_classes: 类别数量，用于构建损失函数。
    
    返回:
    - loss_func: 损失函数，用于计算 ID 损失和三元组损失。
    - center_criterion: 中心损失，用于度量类别中心的距离。
    """
    
    sampler = cfg.DATALOADER.SAMPLER  # 从配置文件中获取数据加载器的采样器类型
    feat_dim = 2048  # 特征维度设置为 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # 中心损失的定义
    
    # 如果模型的度量损失类型包含 'triplet'，则选择三元组损失
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()  # 使用软三元组损失
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # 使用有边界的三元组损失
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))  # 如果度量损失类型不是 'triplet'，则抛出错误信息

    # 如果开启标签平滑正则化
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # 使用标签平滑的交叉熵损失
        print("label smooth on, numclasses:", num_classes)

    # 根据采样器类型定义不同的损失函数
    if sampler == 'softmax':
        def loss_func(score, feat, target):
            # 如果使用 softmax 采样器，只计算交叉熵损失
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            # 如果使用 softmax_triplet 采样器，且度量损失类型为 'triplet'
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                # 标签平滑正则化打开时
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    # 如果 score 是列表，则对每个 score 计算 ID 损失
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)  # 对多个损失进行平均
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)  # 平均与首个损失加权求和
                    else:
                        ID_LOSS = xent(score, target)  # 单一 score 的 ID 损失

                    # 如果 feat 是列表，则对每个 feat 计算三元组损失
                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)  # 对多个损失进行平均
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]  # 平均与首个损失加权求和
                    else:
                            TRI_LOSS = triplet(feat, target)[0]  # 单一特征的三元组损失

                    # 返回 ID 损失与三元组损失的加权和
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    # 标签平滑正则化关闭时
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    # 返回 ID 损失和三元组损失，作为两个独立的值
                    return ID_LOSS, TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))  # 如果度量损失类型不是 'triplet'，则抛出错误信息

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))  # 如果采样器类型不是预期的类型，抛出错误信息
    
    return loss_func, center_criterion  # 返回损失函数和中心损失


def loss_fn_kd(scores, target_scores, T=2., return_score=False):
    """
    计算知识蒸馏（KD）损失，给定学生网络的 [scores] 和教师网络的 [target_scores]。
    
    参数:
    - scores: 学生网络输出的 logits，形状为 (batch_size, num_classes)。
    - target_scores: 教师网络输出的 logits，形状为 (batch_size, num_classes)，通常需要从已有模型中提取。
    - T: 蒸馏温度，控制 softmax 的平滑程度，默认值为 2。
    - return_score: 布尔值，决定是否返回附加信息（最大软标签概率的均值），默认值为 False。

    返回:
    - kd_loss: 知识蒸馏损失。
    - 可选返回：softmax 后的目标概率的最大值的平均值。
    """
    
    device = scores.device  # 获取 scores 所在的设备（CPU 或 GPU）

    # 对学生网络的输出 logits 进行温度缩放后，计算 log_softmax
    log_scores_norm = F.log_softmax(scores / T, dim=1)
    
    # 对教师网络的输出 logits 进行温度缩放后，计算 softmax
    targets_norm = F.softmax(target_scores / T, dim=1)

    # 如果学生网络输出的类别数多于教师网络，给 targets_norm 补零使其维度匹配
    n = scores.size(1)  # 获取学生网络输出的类别数
    if n > target_scores.size(1):
        n_batch = scores.size(0)  # 获取 batch size
        zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))  # 创建补零的张量
        zeros_to_add = zeros_to_add.to(device)  # 将补零张量移动到相同的设备上
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)  # 拼接补零后的张量
    
    # 计算蒸馏损失
    # 使用 KL 散度公式中的交叉熵部分（目标的 softmax 分布与学生网络的 log_softmax 分布）
    kd_loss_unnorm = -(targets_norm * log_scores_norm)
    kd_loss_unnorm = kd_loss_unnorm.sum(dim=1)  # 对类别维度求和
    kd_loss_unnorm = kd_loss_unnorm.mean()  # 对 batch 中的样本求均值

    # 将损失乘以 T^2 进行标准化（标准的知识蒸馏损失计算方式）
    kd_loss = kd_loss_unnorm * T ** 2

    # 如果不需要返回附加信息，只返回蒸馏损失；否则，返回蒸馏损失和 softmax 最大值的均值
    if not return_score:
        return kd_loss
    else:
        return kd_loss, targets_norm.max(dim=1)[0].mean()  # 返回损失和 softmax 后的最大概率值的均值
