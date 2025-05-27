from __future__ import absolute_import

import torch
from ..utils import to_torch


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():  # 禁用梯度计算
        output, target = to_torch(output), to_torch(target)  # 将输出和目标转换为Tensor
        maxk = max(topk)  # 取topk中的最大值
        batch_size = target.size(0)  # 获取批量大小

        _, pred = output.topk(maxk, 1, True, True)  # 获取前maxk个最大值的索引
        pred = pred.t()  # 转置pred，使其可以与目标进行比较
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # 比较预测与目标是否相等

        ret = []
        for k in topk:  # 对于topk中的每个值
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)  # 计算前k个预测正确的数量
            ret.append(correct_k.mul_(1. / batch_size))  # 计算正确的比例，并添加到返回列表中
        return ret  # 返回每个topk对应的准确率
