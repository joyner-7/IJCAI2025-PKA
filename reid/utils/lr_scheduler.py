# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from bisect import bisect_right
import torch
from torch.optim.lr_scheduler import *


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        # 检查 milestones 是否按升序排列
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        # 检查 warmup_method 是否为 'constant' 或 'linear'
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        # 初始化类的属性
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        # 调用父类的构造函数
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    # 定义学习率的计算逻辑
    def get_lr(self):
        warmup_factor = 1
        # 判断当前训练轮数是否小于预热迭代次数
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                # 若为常量预热，直接使用 warmup_factor
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                # 线性预热，根据当前迭代次数与预热迭代次数计算比例因子
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        # 返回每个参数组的学习率，考虑预热因子和步长衰减
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
    

import numpy as np
def warm_up_cosine_lr_scheduler(optimizer, epochs=100, warm_up_epochs=10, eta_min=1e-9):
    """
        Description:
            - Warm up cosin learning rate scheduler, first epoch lr is too small
            - 预热的余弦学习率调度器，在前几个 epoch 中学习率较小
        Arguments:
            - optimizer: input optimizer for the training
              训练时使用的优化器
            - epochs: int, total epochs for your training, default is 100. 
              总训练轮数，默认为100。注意：你需要传递正确的训练轮数
            - warm_up_epochs: int, default is 5, which mean the lr will be warm up for 5 epochs. 
              if warm_up_epochs=0, means no need to warn up, will be as cosine lr scheduler
              预热的 epoch 数，默认是 5，表示学习率会在前 5 个 epoch 内预热。
              如果 warm_up_epochs=0，表示不需要预热，直接使用余弦学习率调度器
            - eta_min: float, setup ConsinAnnealingLR eta_min while warm_up_epochs = 0
              eta_min: 余弦退火学习率调度器的最小学习率，当 warm_up_epochs=0 时有效
        Returns:
            - scheduler
            - 返回调度器
    """
    # 如果不需要预热，直接返回余弦退火学习率调度器
    if warm_up_epochs == 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    else:
        # 定义一个 lambda 函数，表示前 warm_up_epochs 个 epoch 的线性增长，之后使用余弦衰减
        warm_up_with_cosine_lr = lambda epoch: eta_min + (epoch / warm_up_epochs) if epoch < warm_up_epochs else 0.5 * (
            np.cos((epoch - warm_up_epochs) / (epochs - warm_up_epochs) * np.pi) + 1)
        # 返回使用该 lambda 函数作为调度策略的 LambdaLR
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    return scheduler

