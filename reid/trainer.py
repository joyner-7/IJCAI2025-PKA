from __future__ import print_function, absolute_import
import time
from torch.nn import functional as F
import torch
import torch.nn as nn
import tqdm

from reid.loss.loss_uncertrainty import TripletLoss_set
from .utils.meters import AverageMeter
from reid.metric_learning.distance import cosine_similarity

device = torch.device('cuda:0')
        
class Trainer(object):
    def __init__(self, args, model, old_model=None,writer=None):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.writer = writer
        self.uncertainty = True  # 是否使用不确定性损失
        if self.uncertainty:
            self.criterion_triple = TripletLoss_set()  # 三元组损失
        self.criterion_ce = nn.CrossEntropyLoss()  # 交叉熵损失
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')  # KL散度损失
        self.feature_distillation_loss = nn.MSELoss()  # 添加特征蒸馏损失
        self.AF_weight = args.AF_weight  # 反遗忘损失的权重
        self.n_sampling = args.n_sampling  # 采样次数
        
        # 仅在 old_model 存在时保存
        self.old_model = old_model if old_model is not None else None

        #初始化设备
        self.device = torch.device(args.device)
        self.model.to(self.device)
        if self.old_model is not None:
            self.old_model.to(self.device)
        
    def train(self, epoch, data_loader_train, optimizer, training_phase,
            proto_type=None, train_iters=200, add_num=0):

        self.model.train()  # 设置模型为训练模式
        if self.old_model is not None:
            self.old_model.eval()

        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad == False and m.bias.requires_grad == False:
                    m.eval()

        if proto_type is not None:
            print(f"Using proto_type with {len(proto_type)} steps.")
            proto_type_merge = {}
            steps = list(proto_type.keys())
            steps.sort()
            stages = 1
            if stages < len(steps):
                steps = steps[-stages:]

            proto_type_merge['mean_features'] = torch.cat([proto_type[k]['mean_features'] for k in steps])
            proto_type_merge['labels'] = torch.tensor([proto_type[k]['labels'] for k in steps]).to(proto_type_merge['mean_features'].device)

            features_mean = proto_type_merge['mean_features']  # 合并后的原型均值特征

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr1 = AverageMeter()
        losses_tr2 = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            train_inputs = data_loader_train.next()  # 获取训练数据
            data_time.update(time.time() - end)

            s_inputs, targets, cids, domains = self._parse_data(train_inputs)
            targets += add_num  # 调整目标标签

            # 获取模型输出特征
            s_features, merge_feat, cls_outputs, out_var, feat_final_layer = self.model(s_inputs)

            loss_ce, loss_tp1 = 0, 0
            loss_tp2 = torch.tensor(0.0, device=self.device)



            batch_size = s_features.size(0)
            if proto_type is not None:
                steps = sorted(proto_type.keys())[-1:]
                proto_features = torch.cat([proto_type[k]['mean_features'] for k in steps])
                proto_features = F.normalize(proto_features, p=2, dim=1)
                proto_labels = torch.cat([torch.tensor(proto_type[k]['labels']) for k in steps]).to(proto_features.device)
                sampled_prototypes = []
                sampled_labels = []
                num_prototypes_to_sample = batch_size // 2
                selected_indices = torch.randperm(proto_features.size(0))[:num_prototypes_to_sample]
                for idx in selected_indices:
                    # 采样每个原型两次并添加高斯噪声
                    sampled_proto = proto_features[idx].unsqueeze(0)  # 确保每个原型是 [1, 2048]
                    sampled_prototypes.append(sampled_proto)
                    sampled_prototypes.append(sampled_proto)  # 每个原型采样两次

                    sampled_labels.extend([proto_labels[idx].item()] * 2)  # 每个标签对应两次采样

                # 将所有采样的原型拼接成一个 [batch_size, 2048] 的张量
                sampled_prototypes = torch.cat(sampled_prototypes, dim=0)
                sampled_prototypes = self.gaussian_noise(sampled_prototypes, noise_level=0.2)
                sampled_prototypes = F.normalize(sampled_prototypes, p=2, dim=1)  # <--- 添加这一行
                loss_tp2 = torch.tensor(0.0, device=s_features.device)
                loss_tp2 += self.triplet_loss_no_positive(s_features, sampled_prototypes, margin=0.3)
                loss_tp2 = loss_tp2 * 1.5
    
            loss_tp1 = self.criterion_triple(merge_feat, targets)[0]
            loss_tp1 = loss_tp1 * 1.5

            # 计算交叉熵损失
            for s_id in range(1 + self.n_sampling):
                loss_ce += self.criterion_ce(cls_outputs[:, s_id], targets)
            loss_ce = loss_ce / (1 + self.n_sampling)
            loss_ce = loss_ce * 1

            loss = torch.tensor(0.0, device=s_features.device, requires_grad=True)
            if self.uncertainty:
                loss = loss_ce  + loss_tp1 + loss_tp2
            else:
                loss = loss_ce

            losschoice = loss_ce + loss_tp1
            if self.uncertainty:
                losses_tr2.update(loss_tp2.item())
                losses_tr1.update(loss_tp1.item())
            losses_ce.update(loss_ce.item())

            optimizer.zero_grad()
            losschoice.backward(retain_graph=True)

            gradient_scores = {}
            # 计算梯度的绝对值，并获取所有参数的梯度值
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:  # 确保梯度存在
                    grad_score = param.grad.abs().sum().item()  # 计算梯度的绝对值之和
                    gradient_scores[name] = grad_score

            # 按照梯度分数对参数进行排序（从大到小）
            sorted_params = sorted(gradient_scores.items(), key=lambda x: x[1], reverse=True)

            # 选择前 20% 梯度最大的参数
            top_20_percent_count = int(len(sorted_params) * 0.2)
            top_20_percent = sorted_params[:top_20_percent_count]

            # # 打印更新的参数信息
            # print("更新的参数:")
            # for name, score in top_20_percent:
            #     print(f"参数: {name}, 梯度分数: {score:.6f}")

            # 将前 20% 的梯度最大的参数设为 requires_grad = True，其余的设置为 False
            for name, param in self.model.named_parameters():
                # 只更新前 20% 的参数
                if any(name == top_name for top_name, _ in top_20_percent):
                    param.requires_grad = True  # 仅更新梯度分数前 20% 的参数
                else:
                    param.requires_grad = False  # 其余的参数冻结
                    
            loss.backward()
            optimizer.step()

            #恢复所有参数的训练
            for param in self.model.parameters():
                param.requires_grad = True

            batch_time.update(time.time() - end)
            end = time.time()

            if self.writer is not None:
                self.writer.add_scalar(
                    tag=f"loss/Loss_ce_{training_phase}",
                    scalar_value=losses_ce.val,
                    global_step=epoch * train_iters + i
                )
                self.writer.add_scalar(
                    tag=f"loss/Loss_tr1_{training_phase}",
                    scalar_value=losses_tr1.val,
                    global_step=epoch * train_iters + i
                )
                self.writer.add_scalar(
                    tag=f"loss/Loss_tr2_{training_phase}",
                    scalar_value=losses_tr2.val,
                    global_step=epoch * train_iters + i
                )

            if (i + 1) == train_iters:
                print(
                    f"\nEpoch: [{epoch}][{i + 1}/{train_iters}]"
                    f"\n----------------------------------------"
                    f"\nBatch Time     : {batch_time.val:.3f}s (avg: {batch_time.avg:.3f}s)"
                    f"\nData Load Time : {data_time.val:.3f}s (avg: {data_time.avg:.3f}s)"
                    f"\nLoss_ce        : {losses_ce.val:.3f} (avg: {losses_ce.avg:.3f})"
                    f"\nLoss_tp1       : {losses_tr1.val:.3f} (avg: {losses_tr1.avg:.3f})"
                    f"\nLoss_tp2       : {losses_tr2.val:.3f} (avg: {losses_tr2.avg:.3f})"
                    f"\n----------------------------------------"
                    f"\nTotal Loss     : {loss.item():.3f}"
                )

    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix
    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.to(device)  # 将输入张量移动到主设备
        targets = pids.to(device)  # 将目标张量移动到主设备
        #inputs = imgs.cuda()
        #targets = pids.cuda()
        return inputs, targets, cids, domains
    
    def gaussian_sample(self,proto_features, n_samples):
        """
        从 proto_features 中进行高斯采样
        :param proto_features: 原型特征 (C, feature_dim)
        :param n_samples: 每个原型要采样的数量
        :return: 采样的特征
        """ 
        # 获取原型特征的维度
        C, feature_dim = proto_features.size()
        # 生成随机高斯噪声
        noise = torch.randn(C * n_samples, feature_dim).to(proto_features.device) * 0.2   # 0.1 是噪声的标准差
        # 对每个原型特征添加高斯噪声
        sampled_prototypes = proto_features.repeat(n_samples, 1) + noise
        return sampled_prototypes
    # 在 Trainer 类中添加这个新方法
    def gaussian_noise(self, features, noise_level=0.1):
        """
        给输入的特征张量添加高斯噪声。
        :param features: 特征张量 (N, feature_dim)
        :param noise_level: 噪声的标准差
        :return: 添加了噪声的特征张量
        """
        noise = torch.randn_like(features) * noise_level
        return features + noise
    
    # def triplet_loss_no_positive(self, anchor, negatives, margin=0.3):
    #     """
    #     计算变种三元组损失，仅包含 Anchor 和 Negative。
    #     :param anchor: 锚点特征 (Tensor) [batch_size, feature_dim]
    #     :param negatives: 负样本特征 (Tensor) [batch_size, feature_dim]
    #     :param margin: 边际值 (float)，默认值为 0.3。
    #     :return: 损失值 (Tensor)
    #     """
    #     # 计算锚点与负样本的欧几里得距离
    #     dist_neg = torch.norm(anchor.unsqueeze(1) - negatives, p=2, dim=2)  # [batch_size, num_negatives]

    #     # 按最小负样本距离计算损失（即最近的负样本）
    #     min_dist_neg, _ = torch.min(dist_neg, dim=1)  # [batch_size]

    #     # 计算三元组损失: max(0, margin - min_dist_neg)
    #     loss = F.relu(margin - min_dist_neg)

    #     # 返回平均损失    
    #     return loss.mean()

    # def triplet_loss_no_positive(self, anchor, negatives, margin=0.3):
    #     dist_neg = torch.norm(anchor.unsqueeze(1) - negatives, p=2, dim=2)
    #     min_dist_neg, _ = torch.min(dist_neg, dim=1)
    #     loss = F.relu(margin - min_dist_neg)
    #     # print(f"  INSIDE triplet_loss_no_positive: margin={margin:.3f}, "
    #     #       f"min_dist_neg mean={min_dist_neg.mean().item():.3f}, "
    #     #       f"min_dist_neg min={min_dist_neg.min().item():.3f}, "
    #     #       f"max={min_dist_neg.max().item():.3f}, "
    #     #       f"loss_before_mean={loss.mean().item():.3f} " # 实际上 loss.mean() 就是返回值
    #     #       #f"min_dist_neg values (first 3): {min_dist_neg[:3].tolist()}"
    #     #       )
    #     return loss.mean()
    
    def triplet_loss_no_positive(self, anchor, negative, margin=0.3):
        """修改后的三元组损失函数，没有正样本，仅推远负样本"""
        # 计算负样本与锚点之间的欧氏距离
        neg_distance = F.pairwise_distance(anchor, negative, 2)  
        loss = F.relu(neg_distance + margin)  # 只需要推远负样本
        return loss.mean()  # 返回标量 Tensor
    

    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, cids, domains



