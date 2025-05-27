from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    """
    返回列表 a 中所有元素不等于 b 的索引列表。
    
    参数:
    - a: 输入列表。
    - b: 要排除的值。
    
    返回:
    - 列表 a 中不等于 b 的元素的索引。
    """
    assert isinstance(a, list)  # 确保输入 a 是一个列表
    return [i for i, j in enumerate(a) if j != b]  # 返回 a 中所有不等于 b 的元素索引


class RandomIdentitySampler(Sampler):
    """
    随机身份采样器（RandomIdentitySampler），用于在 re-id 任务中，
    从每个身份（person ID, pid）中随机采样若干个实例（图片）。

    参数:
    - data_source: 数据源，通常是一个包含数据元组（图片路径，person ID，摄像头 ID，时间戳）的列表。
    - num_instances: 每个身份（pid）要采样的实例数量。
    """
    
    def __init__(self, data_source, num_instances):
        self.data_source = data_source  # 数据集
        self.num_instances = num_instances  # 每个身份要采样的实例数量
        self.index_dic = defaultdict(list)  # 字典，键为 pid，值为该 pid 对应的数据索引列表

        # 遍历数据源，将每个数据按 pid 分组，存储到 index_dic 中
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)  # 将当前数据的索引添加到对应 pid 的列表中

        self.pids = list(self.index_dic.keys())  # 获取所有身份的 pid 列表
        self.num_samples = len(self.pids)  # 总共的身份数
    
    def __len__(self):
        """
        返回总样本数：num_samples * num_instances。
        这是所有身份的数量乘以每个身份要采样的实例数。
        """
        return self.num_samples * self.num_instances
    
    def __iter__(self):
        """
        迭代器，返回打乱顺序后的索引列表，每个身份返回指定数量的实例。
        """
        indices = torch.randperm(self.num_samples).tolist()  # 随机打乱所有身份的顺序，得到一个随机身份索引列表
        ret = []  # 用于存储采样的索引结果

        # 遍历每个身份的随机索引
        for i in indices:
            pid = self.pids[i]  # 获取当前身份 pid
            t = self.index_dic[pid]  # 获取当前身份 pid 对应的所有数据索引列表

            # 如果该身份的实例数大于等于 num_instances，随机选取 num_instances 个实例
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                # 否则，使用有放回抽样，保证实例数达到 num_instances
                t = np.random.choice(t, size=self.num_instances, replace=True)
            
            ret.extend(t)  # 将采样到的实例索引添加到返回列表中
        
        return iter(ret)  # 返回迭代器，迭代输出采样到的索引

class MultiDomainRandomIdentitySampler(Sampler):
    """
    多域随机身份采样器，适用于跨多个域的数据集的采样。
    该采样器在每个域中随机采样身份，并从每个身份中随机选择实例。
    
    参数:
    - data_source: 数据源，包含数据项（如图像路径、身份ID、摄像头ID、域ID等）的列表。
    - num_instances: 每个身份需要采样的实例数量。
    """
    
    def __init__(self, data_source, num_instances):
        self.data_source = data_source  # 数据集
        self.num_instances = num_instances  # 每个身份采样的实例数

        # 初始化两个字典：
        # domain2pids: 按域组织的身份ID列表
        # pid2index: 每个身份ID对应的数据索引列表
        self.domain2pids = defaultdict(list)
        self.pid2index = defaultdict(list)

        # 遍历数据源，将每条数据按域和身份进行分类
        for index, (_, pid, _, domain) in enumerate(data_source):
            if pid not in self.domain2pids[domain]:
                self.domain2pids[domain].append(pid)
            self.pid2index[pid].append(index)  # 记录每个身份对应的索引

        self.pids = list(self.pid2index.keys())  # 获取所有身份ID
        self.domains = list(sorted(self.domain2pids.keys()))  # 获取所有域ID并排序

        self.num_samples = len(self.pids)  # 样本总数

    def __len__(self):
        return self.num_samples * self.num_instances  # 总样本数 = 身份数 * 每个身份的实例数

    def __iter__(self):
        """
        迭代器，返回打乱顺序后的采样索引。
        """
        ret = []  # 存储最终的采样结果
        domain2pids = copy.deepcopy(self.domain2pids)  # 深拷贝域-身份字典，防止修改原始数据

        # 重复采样，假设每轮8次
        for _ in range(8):
            for domain in self.domains:  # 遍历每个域
                pids = np.random.choice(domain2pids[domain], size=8, replace=False)  # 从每个域随机选8个身份
                for pid in pids:
                    idxs = copy.deepcopy(self.pid2index[pid])  # 获取该身份的所有索引
                    idxs = np.random.choice(idxs, size=2, replace=False)  # 从该身份中随机选2个索引
                    ret.extend(idxs)  # 将采样的索引添加到返回结果中

        return iter(ret)  # 返回索引的迭代器


class RandomMultipleGallerySampler(Sampler):
    """
    多样本采样器，用于从一个身份的多个摄像头视角或多个实例中采样数据。
    
    参数:
    - data_source: 数据源，包含数据项（如图像路径、身份ID、摄像头ID等）的列表。
    - num_instances: 每个身份采样的实例数量，默认值为4。
    """
    
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source  # 数据集
        self.index_pid = defaultdict(int)  # 记录每个索引对应的身份ID
        self.pid_cam = defaultdict(list)  # 记录每个身份对应的摄像头ID列表
        self.pid_index = defaultdict(list)  # 记录每个身份对应的数据索引列表
        self.num_instances = num_instances  # 每个身份采样的实例数

        # 遍历数据源，将每条数据按身份和摄像头ID进行分类
        try:
            for index, (_, pid, cam, frame) in enumerate(data_source):
                self.index_pid[index] = pid  # 记录每个索引的身份ID
                self.pid_cam[pid].append(cam)  # 记录每个身份的摄像头ID
                self.pid_index[pid].append(index)  # 记录每个身份的索引
        except:
            # 兼容不同的数据格式
            for index, (_, pid, cam, _, _) in enumerate(data_source):
                self.index_pid[index] = pid
                self.pid_cam[pid].append(cam)
                self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())  # 获取所有身份ID
        self.num_samples = len(self.pids)  # 样本总数

    def __len__(self):
        return self.num_samples * self.num_instances  # 总样本数 = 身份数 * 每个身份的实例数

    def __iter__(self):
        """
        迭代器，返回打乱顺序后的采样索引。
        """
        indices = torch.randperm(len(self.pids)).tolist()  # 随机打乱身份ID顺序
        ret = []  # 存储最终的采样结果

        # 遍历随机打乱的身份ID
        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])  # 从该身份中随机选取一个索引
            _, i_pid, i_cam, _ = self.data_source[i]  # 获取选中索引的身份ID和摄像头ID

            ret.append(i)  # 添加该索引到返回结果中

            pid_i = self.index_pid[i]  # 获取当前索引对应的身份ID
            cams = self.pid_cam[pid_i]  # 获取该身份对应的所有摄像头ID
            index = self.pid_index[pid_i]  # 获取该身份对应的所有数据索引
            select_cams = No_index(cams, i_cam)  # 获取该身份其他摄像头ID

            if select_cams:
                # 如果有其他摄像头ID，从中随机选取（num_instances - 1）个
                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)
                for kk in cam_indexes:
                    ret.append(index[kk])  # 将选中的索引添加到返回结果中
            else:
                # 如果没有其他摄像头ID，随机选择其他索引
                select_indexes = No_index(index, i)
                if not select_indexes: continue  # 如果没有可选的其他索引，跳过
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)
                for kk in ind_indexes:
                    ret.append(index[kk])  # 将选中的索引添加到返回结果中

        return iter(ret)  # 返回索引的迭代器