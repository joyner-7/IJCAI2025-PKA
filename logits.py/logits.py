from __future__ import print_function, absolute_import
import time
from torch.nn import functional as F
import torch
import torch.nn as nn

from reid.loss.loss_uncertrainty import TripletLoss_set
from .utils.meters import AverageMeter
from reid.metric_learning.distance import cosine_similarity

class Trainer(object):
    def __init__(self, args, model, old_model=None, writer=None):
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

    def train(self, epoch, data_loader_train, optimizer, training_phase,
              proto_type=None, train_iters=200, add_num=0):

        self.model.train()  # 设置模型为训练模式
        if self.old_model is not None:
            self.old_model.eval()  # 旧模型作为教师模型

        # 冻结批归一化层
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if not m.weight.requires_grad and not m.bias.requires_grad:
                    m.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr1 = AverageMeter()
        losses_logit_dist = AverageMeter()  # 新增：记录逻辑蒸馏损失

        end = time.time()

        for i in range(train_iters):
            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)

            s_inputs, targets, cids, domains = self._parse_data(train_inputs)
            targets += add_num

            # 获取模型输出
            s_features, merge_feat, cls_outputs, out_var, feat_final_layer = self.model(s_inputs)

            # 获取教师模型的 logits（仅在 old_model 存在时）
            if self.old_model is not None:
                with torch.no_grad():
                    teacher_logits = self.old_model(s_inputs)[1]  # 教师模型的分类输出（假设是 logits）
                    print(f"teacher_logits shape: {teacher_logits.shape}")
                    print(f"cls_outputs shape: {cls_outputs.shape}")

            # 初始化损失
            loss_ce, loss_tp1, loss_logit_dist = 0, 0, 0

            # 计算三元组损失
            loss_tp1 = self.criterion_triple(merge_feat, targets)[0] * 1.5

            # 计算交叉熵损失
            for s_id in range(1 + self.n_sampling):
                loss_ce += self.criterion_ce(cls_outputs[:, s_id], targets)
            loss_ce /= (1 + self.n_sampling)

            # # 计算逻辑蒸馏损失（仅在 old_model 存在时）
            # if self.old_model is not None:
            #     temperature = 2.0  # 温度参数

            #     # 只取新模型输出的前一部分，与旧模型的输出维度对齐
            #     min_classes = min(teacher_logits.size(2), cls_outputs.size(2))
            #         # 打印 logits 的形状
            #     # print(f"teacher_logits shape: {teacher_logits.shape}")  # 教师模型的输出形状
            #     # print(f"cls_outputs shape: {cls_outputs.shape}")  # 学生模型的输出形状
            #     # print(f"min_classes: {min_classes}")  # 最小类数

            #     student_probs = F.log_softmax(cls_outputs[:, :, :min_classes] / temperature, dim=2)  # 
            #     teacher_probs = F.softmax(teacher_logits / temperature, dim=2)  # 保持教师的 logits 不变
            #     # 打印学生和教师概率的形状
            #     # print(f"student_probs shape: {student_probs.shape}")  # 学生模型的概率形状
            #     # print(f"teacher_probs shape: {teacher_probs.shape}")  # 教师模型的概率形状

            #     # 计算KL散度损失
            #     loss_logit_dist = (
            #         self.KLDivLoss(student_probs, teacher_probs) * (temperature ** 2)
            #     )

            # 汇总总损失
            loss = loss_ce + loss_tp1
            if self.old_model is not None:
                loss += loss_logit_dist  # 加入逻辑蒸馏损失

            # 更新损失监控
            losses_ce.update(loss_ce.item())
            losses_tr1.update(loss_tp1.item())
            if self.old_model is not None:
                losses_logit_dist.update(loss_logit_dist.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        if self.writer is not None:
            # 将所有损失添加到 TensorBoard 中
            self.writer.add_scalar(
                tag=f"loss/Loss_ce_{training_phase}",
                scalar_value=losses_ce.val,
                global_step=epoch * train_iters + i,
            )
            self.writer.add_scalar(
                tag=f"loss/Loss_tr1_{training_phase}",
                scalar_value=losses_tr1.val,
                global_step=epoch * train_iters + i,
            )
            if self.old_model is not None:
                self.writer.add_scalar(
                    tag=f"loss/Loss_logit_dist_{training_phase}",
                    scalar_value=losses_logit_dist.val,
                    global_step=epoch * train_iters + i,
                )

        # 每个 epoch 结束时打印详细的损失信息
        if (i + 1) == train_iters:
            print(
                f"\nEpoch: [{epoch}][{i + 1}/{train_iters}]"
                f"\n----------------------------------------"
                f"\nBatch Time     : {batch_time.val:.3f}s (avg: {batch_time.avg:.3f}s)"
                f"\nData Load Time : {data_time.val:.3f}s (avg: {data_time.avg:.3f}s)"
                f"\nLoss_ce        : {losses_ce.val:.3f} (avg: {losses_ce.avg:.3f})"
                f"\nLoss_tp1       : {losses_tr1.val:.3f} (avg: {losses_tr1.avg:.3f})"
                f"\nLoss_logit_dist: {losses_logit_dist.val:.3f} (avg: {losses_logit_dist.avg:.3f})"
                f"\n----------------------------------------"
                f"\nTotal Loss     : {loss.item():.3f}"
            )

    def get_normal_affinity(self, x, Norm=0.1):
        pre_matrix_origin = cosine_similarity(x, x)
        pre_affinity_matrix = F.softmax(pre_matrix_origin / Norm, dim=1)
        return pre_affinity_matrix

    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        device = torch.device('cuda:0')
        inputs = imgs.to(device)  # 将输入张量移动到主设备
        targets = pids.to(device)  # 将目标张量移动到主设备
        return inputs, targets, cids, domains

    def gaussian_sample(self, proto_features, n_samples):
        C, feature_dim = proto_features.size()
        noise = torch.randn(C * n_samples, feature_dim).to(proto_features.device) * 0.1
        sampled_prototypes = proto_features.repeat(n_samples, 1) + noise
        return sampled_prototypes
