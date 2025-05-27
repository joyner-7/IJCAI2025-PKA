import copy

import torch.nn as nn
import torchvision.models as models

import torchvision
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

class GeneralizedMeanPooling(nn.Module):
    r"""应用2D幂平均自适应池化到由多个输入平面组成的输入信号上。
    计算函数为： :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - 当 p = 无穷大时，得到最大池化
        - 当 p = 1 时，得到平均池化
    输出的大小为 H x W，适用于任何输入大小。
    输出特征的数量等于输入平面的数量。
    参数:
        output_size: 目标输出大小，形状为 H x W。
                     可以是元组 (H, W) 或单个 H 表示方形图像 H x H
                     H 和 W 可以是 ``int`` 类型，或者 ``None``，表示大小与输入相同。
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)  # 幂参数
        self.output_size = output_size  # 输出大小
        self.eps = eps  # 避免数值计算中的零值

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)  # 避免零值并计算幂
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)  # 自适应池化并计算幂的倒数

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)


class Normalize(nn.Module):
    def __init__(self, power=2, dim=1):
        super(Normalize, self).__init__()
        self.power = power  # 正则化的幂
        self.dim = dim  # 规范化的维度

    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim, keepdim=True).pow(1. / self.power)  # 计算范数
        out = x.div(norm + 1e-4)  # 进行规范化
        return out


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model='resnet50', out_dim=2048, n_sampling=2, pool_len=8, normal_feature=True,
                 num_classes=-1, uncertainty=True):
        super(ResNetSimCLR, self).__init__()

        # 定义不同的 ResNet 模型
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=True)}
        self.resnet = self._get_basemodel(base_model)  # 获取基础模型
        self.base = nn.Sequential(*list(self.resnet.children())[:-3])  # 去掉最后几层
        dim_mlp = self.resnet.fc.in_features // 2  # 计算全连接层的输入维度
        self.linear_mean = nn.Linear(dim_mlp, out_dim)  # 用于计算均值的线性层
        self.linear_var = nn.Linear(dim_mlp, out_dim)  # 用于计算方差的线性层
        self.pool_len = 8  # 池化层的长度
        self.conv_var = nn.Conv2d(dim_mlp, dim_mlp, kernel_size=(pool_len, pool_len), bias=False)  # 用于计算方差的卷积层

        self.n_sampling = n_sampling  # 采样次数
        self.n_samples = torch.Size(np.array([n_sampling, ]))  # 采样尺寸
        self.pooling_layer = GeneralizedMeanPoolingP(3)  # 定义池化层

        self.l2norm_mean, self.l2norm_var, self.l2norm_sample = Normalize(2, 1), Normalize(2, 1), Normalize(2, 2)  # 归一化层

        print('using resnet50 as a backbone')  # 打印使用的模型
        '''xkl add'''
        print("##########normalize matchiing feature:", normal_feature)  # 打印是否匹配特征
        self.normal_feature = normal_feature  # 是否规范化特征
        self.uncertainty = uncertainty  # 是否使用不确定性

        self.bottleneck = nn.BatchNorm2d(out_dim)  # 批量归一化层
        self.bottleneck.bias.requires_grad_(False)  # 不需要训练偏置
        nn.init.constant_(self.bottleneck.weight, 1)  # 初始化权重为1
        nn.init.constant_(self.bottleneck.bias, 0)  # 初始化偏置为0

        self.classifier = nn.Linear(out_dim, num_classes, bias=False)  # 分类器
        nn.init.normal_(self.classifier.weight, std=0.001)  # 初始化分类器权重
        self.relu = nn.ReLU()  # 激活函数

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]  # 根据名称获取模型
        return model

    # def forward(self, x, training_phase=None, fkd=False):
    #     BS = x.size(0)  # 批量大小
        
    #     out = self.base(x)  # 前向传播到基础模型
    #     out_mean = self.pooling_layer(out)  # 全局池化
    #     out_mean = out_mean.view(out_mean.size(0), -1)  # B x 1024
    #     out_mean = self.linear_mean(out_mean)  # Bx2048
    #     # out_mean = self.l2norm_mean(out_mean)  # L2归一化

    #     out_var = self.conv_var(out)  # 卷积层
    #     out_var = self.pooling_layer(out_var)  # 池化
    #     out_var += 1e-4  # 加一个小常数
    #     out_var = out_var.view(out_var.size(0), -1)  # Bx1024
    #     out_var = self.linear_var(out_var)  # Bx2049

    #     out_mean = self.l2norm_mean(out_mean)  # 对均值特征进行L2归一化

    #     var_choice = 'L2'  # 选择方差处理方式为'L2'
    #     if var_choice == 'L2':
    #         out_var = self.l2norm_var(out_var)  # 对方差特征进行L2归一化
    #         out_var = self.relu(out_var) + 1e-4  # 使用ReLU激活，并加上小的常数防止数值问题
    #     elif var_choice == 'softmax':
    #         out_var = F.softmax(out_var, dim=1)  # 对方差特征进行softmax处理
    #         out_var = out_var.clone()  # 克隆数据，防止反向传播时出现梯度计算错误
    #     elif var_choice == 'log':
    #         out_var = torch.exp(0.5 * out_var)  # 对方差特征进行指数计算处理

    #     if self.uncertainty:  # 如果启用不确定性估计
    #         BS, D = out_mean.size()  # 获取批次大小和特征维度
    #         tdist = MultivariateNormal(loc=out_mean, scale_tril=torch.diag_embed(out_var))  # 创建多变量正态分布
    #         samples = tdist.rsample(self.n_samples)  # 从分布中采样 (n_samples, batch_size, out_dim)

    #         samples = self.l2norm_sample(samples)  # 对采样的特征进行L2归一化

    #         # 将均值特征与采样的特征合并 (n_samples+1, batch_size, out_dim)
    #         merge_feat = torch.cat((out_mean.unsqueeze(0), samples), dim=0)  
    #         merge_feat = merge_feat.resize(merge_feat.size(0) * merge_feat.size(1), merge_feat.size(-1))  # 重塑为 ((n_samples+1) * batch_size, out_dim)
            
    #         # 通过瓶颈层 (BatchNorm)
    #         bn_feat = self.bottleneck(merge_feat.unsqueeze(-1).unsqueeze(-1))  # [(n_samples+1) * batch_size, out_dim, 1, 1]
            
    #         # 分类输出
    #         cls_outputs = self.classifier(bn_feat[..., 0, 0])  # [(n_samples+1) * batch_size, num_classes]
            
    #         # 重新调整为 (n_samples+1, batch_size, out_dim)
    #         merge_feat = merge_feat.resize(self.n_sampling + 1, BS, merge_feat.size(-1))  
    #         cls_outputs = cls_outputs.resize(self.n_sampling + 1, BS, cls_outputs.size(-1))  # (n_samples+1, batch_size, num_classes)
    #     else:  
    #         # 如果不启用不确定性估计，直接对均值特征进行瓶颈处理和分类
    #         bn_feat = self.bottleneck(out_mean.unsqueeze(-1).unsqueeze(-1))  # [batch_size, 2048, 1, 1]
    #         cls_outputs = self.classifier(bn_feat[..., 0, 0])  # [batch_size, num_classes]
    #         #cls_outputs = cls_outputs.unsqueeze(0)  # [1, batch_size, num_classes]
    #         cls_outputs = cls_outputs.resize(self.n_sampling + 1, BS, cls_outputs.size(-1))  # (n_samples+1, batch_size, num_classes)
    #         merge_feat = out_mean.unsqueeze(0)  # [1, batch_size, 2048]

    #     if fkd:  # 如果启用知识蒸馏 (feature knowledge distillation)，返回所有特征
    #         return out_mean, merge_feat.permute(1, 0, 2), cls_outputs.permute(1, 0, 2), out_var, out

    #     if self.training:  # 如果处于训练模式，返回所有特征
    #         #均值特征、合并特征、分类输出、方差特征和原始特征。
    #         return out_mean, merge_feat.permute(1, 0, 2), cls_outputs.permute(1, 0, 2), out_var, out
    #     else:  # 如果处于评估模式，仅返回均值特征
    #         return out_mean[:BS], cls_outputs.permute(1, 0, 2)

    def forward(self, x, training_phase=None, fkd=False):
        BS = x.size(0)  # 获取批量大小
        out = self.base(x)  # 通过基础网络层 (去除ResNet最后几层的主干网络)
        
        # 计算均值特征路径
        out_mean = self.pooling_layer(out)  # 对 'out' 进行全局池化 (GeneralizedMeanPoolingP)
        out_mean = out_mean.view(out_mean.size(0), -1)  # 将池化后的特征展平，形状变为 (BS, D_pooled)
        out_mean = self.linear_mean(out_mean)  # 通过线性层得到初步的均值特征 (BS, out_dim)

        # 计算方差特征路径 (这里的方差更像是用于生成不确定性的特征，而不是直接的统计方差)
        out_var_feat = self.conv_var(out)  # 对 'out' 使用特定的卷积层 (self.conv_var)
        out_var_feat = self.pooling_layer(out_var_feat)  # 对卷积后的特征进行全局池化
        out_var_feat += 1e-4  # 添加一个小的epsilon，防止数值问题
        out_var_feat = out_var_feat.view(out_var_feat.size(0), -1)  # 将池化后的方差特征展平
        out_var_feat = self.linear_var(out_var_feat)  # 通过线性层得到初步的方差相关特征

        # 对均值特征进行L2归一化，得到最终的 s_features (通常用于ReID匹配的特征)
        s_features = self.l2norm_mean(out_mean)  # (BS, out_dim)，L2范数为1

        # 对初步的方差特征进行处理，得到最终的 out_var (用于多变量正态分布的scale_tril对角线)
        var_choice = 'L2'  # 方差处理方式选择
        if var_choice == 'L2':
            out_var = self.l2norm_var(out_var_feat)  # 对初步方差特征进行L2归一化
            out_var = self.relu(out_var) + 1e-4    # 通过ReLU激活并加epsilon (确保为正，作为标准差或方差的代理)
        elif var_choice == 'softmax':
            out_var = F.softmax(out_var_feat, dim=1) # 使用softmax处理
            out_var = out_var.clone()                 # 克隆以避免潜在的inplace修改问题
        elif var_choice == 'log':
            out_var = torch.exp(0.5 * out_var_feat)   # 假设 out_var_feat 是 log(variance)，得到标准差

        # 初始化最终返回的合并特征、分类输出和最后一层特征
        final_merge_feat = None
        final_cls_outputs = None
        final_feat_final_layer = out  # 'out' 是 self.base 的输出，即池化前的深层特征图

        if self.uncertainty:  # 如果启用不确定性估计
            BS_current, D = s_features.size()  # 获取当前批次大小和特征维度
            # 创建多变量正态分布，均值为s_features，协方差矩阵的对角线为out_var (解释为每个维度的方差或标准差的平方)
            # torch.diag_embed(out_var) 创建一个对角矩阵，对角线元素是 out_var 的值
            tdist = MultivariateNormal(loc=s_features, scale_tril=torch.diag_embed(out_var))
            samples = tdist.rsample(self.n_samples)  # 从分布中进行可重参数化采样，形状 (n_samples, BS_current, D)
            samples = self.l2norm_sample(samples)  # 对采样的特征进行L2归一化，在第2维上 (特征维度)

            # 将归一化的均值特征 (s_features) 与归一化的采样特征 (samples) 合并
            # s_features.unsqueeze(0) 形状 (1, BS_current, D)
            # current_merge_feat 形状 (n_samples + 1, BS_current, D)
            current_merge_feat = torch.cat((s_features.unsqueeze(0), samples), dim=0)
            # 将合并后的特征展平，以便通过后续层
            # 形状 ((n_samples + 1) * BS_current, D)
            current_merge_feat_flat = current_merge_feat.reshape(current_merge_feat.size(0) * current_merge_feat.size(1), current_merge_feat.size(-1))

            # 通过瓶颈层 (BatchNorm2d，所以需要 unsqueeze 添加通道和空间维度)
            # bn_feat 形状 ((n_samples + 1) * BS_current, D, 1, 1)
            bn_feat = self.bottleneck(current_merge_feat_flat.unsqueeze(-1).unsqueeze(-1))
            # 对瓶颈层输出进行分类
            # bn_feat[..., 0, 0] 形状 ((n_samples + 1) * BS_current, D)
            # current_cls_outputs_flat 形状 ((n_samples + 1) * BS_current, num_classes)
            current_cls_outputs_flat = self.classifier(bn_feat[..., 0, 0])

            # 将合并特征和分类输出的形状调整回 (BS_current, n_samples + 1, FeatureDim/NumClasses) 以匹配 Trainer 中的期望
            # permute(1,0,2) 将 (n_samples + 1, BS_current, Dim) 变为 (BS_current, n_samples + 1, Dim)
            final_merge_feat = current_merge_feat_flat.reshape(self.n_sampling + 1, BS_current, current_merge_feat_flat.size(-1)).permute(1,0,2)
            final_cls_outputs = current_cls_outputs_flat.reshape(self.n_sampling + 1, BS_current, current_cls_outputs_flat.size(-1)).permute(1,0,2)
        else:  # 如果不启用不确定性估计
            # 对归一化的均值特征 (s_features) 进行瓶颈处理和分类
            bn_feat = self.bottleneck(s_features.unsqueeze(-1).unsqueeze(-1)) # (BS, D, 1, 1)
            current_cls_outputs = self.classifier(bn_feat[..., 0, 0]) # (BS, num_classes)

            # 为了与不确定性情况下的输出形状保持一致（主要是为了下游如Trainer的统一处理）
            # 将分类输出和均值特征在第1维（采样维）上复制 n_sampling + 1 次
            # final_cls_outputs 形状 (BS, n_sampling + 1, num_classes)
            final_cls_outputs = current_cls_outputs.unsqueeze(1).repeat(1, self.n_sampling + 1, 1)
            # final_merge_feat 形状 (BS, n_sampling + 1, D)
            final_merge_feat = s_features.unsqueeze(1).repeat(1, self.n_sampling + 1, 1)


        # 根据 fkd (feature knowledge distillation) 标志决定返回值
        # 如果 fkd 为 True，通常用于知识蒸馏，返回所有5个值
        if fkd:
             return s_features, final_merge_feat, final_cls_outputs, out_var, final_feat_final_layer

        # 无论是在训练模式还是评估模式 (当 fkd 不为 True 时)，
        # 都返回标准的5个输出，以确保 `extract_features_uncertain` 和 `Trainer.train` 能获取所需的值。
        # `s_features` 是L2归一化的均值特征。
        # `final_merge_feat` 是合并后的特征，形状 (BS, n_sampling+1, D)。
        # `final_cls_outputs` 是分类输出，形状 (BS, n_sampling+1, num_classes)。
        # `out_var` 是处理后的方差相关特征。
        # `final_feat_final_layer` 是 `self.base` 的输出（池化前的特征图）。
        return s_features, final_merge_feat, final_cls_outputs, out_var, final_feat_final_layer


if __name__ == '__main__':
    m = ResNetSimCLR(uncertainty=True)  # 实例化模型，并启用不确定性估计
    m(torch.zeros(10, 3, 256, 128))  # 使用一个形状为 (10, 3, 256, 128) 的全零张量进行前向传播测试

