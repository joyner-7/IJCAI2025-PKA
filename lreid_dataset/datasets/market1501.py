from __future__ import division, print_function, absolute_import
import os
import copy
from lreid_dataset.incremental_datasets import IncrementalPersonReIDSamples
# from reid.utils.serialization import read_json, write_json
# from lreid_dataset.datasets import ImageDataset
import re
import glob
import os.path as osp
import warnings

class IncrementalSamples4market(IncrementalPersonReIDSamples):
    '''
    Market 数据集处理类
    '''
    _junk_pids = [0, -1]  # 表示垃圾图像的 ID 列表
    dataset_dir = 'market1501'  # 数据集的目录名称
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'  # 数据集下载 URL

    def __init__(self, datasets_root, relabel=True, combineall=False):
        # 初始化方法，设置根目录、是否重新标记 ID 以及是否合并所有数据
        self.relabel = relabel  # 是否重新标记 ID
        self.combineall = combineall  # 是否合并所有训练数据（包括查询和库）
        root = osp.join(datasets_root, self.dataset_dir, 'Market-1501-v15.09.15')  
        # 构建数据集的根路径
        self.train_dir = osp.join(root, 'bounding_box_train')  # 训练集目录
        self.query_dir = osp.join(root, 'query')  # 查询集目录
        self.gallery_dir = osp.join(root, 'bounding_box_test')  # 库集目录
        train = self.process_dir(self.train_dir, relabel=True)  # 处理训练集，并重新标记 ID
        query = self.process_dir(self.query_dir, relabel=False)  # 处理查询集，不重新标记 ID
        gallery = self.process_dir(self.gallery_dir, relabel=False)  # 处理库集，不重新标记 ID
        self.train, self.query, self.gallery = train, query, gallery  # 将处理后的数据集赋值给类属性
        self._show_info(train, query, gallery)  # 显示数据集的基本信息

    def process_dir(self, dir_path, relabel=False):
        # 处理数据集文件夹中的图像，并返回包含图像路径、行人 ID 和摄像头 ID 的列表
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))  # 获取文件夹中所有图片的路径
        pattern = re.compile(r'([-\d]+)_c(\d)')  # 定义用于提取行人 ID 和摄像头 ID 的正则表达式

        pid_container = set()  # 创建一个集合，用于存储所有行人 ID
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())  # 从图片文件名中提取行人 ID 和摄像头 ID
            if pid == -1:
                continue  # 忽略垃圾图像（行人 ID 为 -1）
            pid_container.add(pid)  # 将行人 ID 添加到集合中
        pid2label = {pid: label for label, pid in enumerate(pid_container)}  # 为每个行人 ID 分配一个新的标签（从 0 开始）

        data = []  # 用于存储处理后的图像数据
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())  # 提取图片的行人 ID 和摄像头 ID
            if pid == -1:
                continue  # 忽略垃圾图像
            assert 0 <= pid <= 1501  # 确保行人 ID 合法（0 表示背景，1-1501 为有效行人 ID）
            assert 1 <= camid <= 6  # 确保摄像头 ID 合法（1-6 表示有效摄像头）
            camid -= 1  # 摄像头 ID 从 0 开始计数
            if relabel:
                pid = pid2label[pid]  # 如果需要重新标记 ID，则使用新的标签替换原 ID
            data.append((img_path, pid, camid, 0))  # 将图像路径、行人 ID、摄像头 ID 以及第 4 列的数据（0）添加到列表中

        return data  # 返回处理后的图像数据



class Market1501(IncrementalPersonReIDSamples):
    """Market1501 数据集处理类.

    参考文献:
        Zheng 等人. 可扩展的行人重识别：一个基准测试. ICCV 2015.

    数据集统计信息:
        - 行人身份: 1501（+1 表示背景）。
        - 图片数量: 12936（训练）+ 3368（查询）+ 15913（库）。
    """
    _junk_pids = [0, -1]  # 表示垃圾图片的 ID 列表
    dataset_dir = 'market1501'  # 数据集目录名称
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'  # 数据集下载链接

    def __init__(self, root='/data0/data_lzj', market1501_500k=False, **kwargs):
        # 初始化方法，定义数据集根目录及其他可选参数
        self.root = osp.expanduser(root)  # 扩展用户路径，获取数据集根目录
        self.dataset_dir = osp.join(self.root, self.dataset_dir, 'Market-1501-v15.09.15')  
        # 数据集路径
        self.download_dataset(self.dataset_dir, self.dataset_url)  # 下载数据集

        # 允许替代的数据集目录结构
        self.data_dir = self.dataset_dir  # 数据目录
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')  # 数据子目录
        if osp.isdir(data_dir):  # 检查是否为有效目录
            self.data_dir = data_dir  # 更新数据目录
        else:
            warnings.warn(
                '当前数据结构已弃用。请将“bounding_box_train”等文件夹放置在'
                '"Market-1501-v15.09.15"目录下。'
            )

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')  # 训练集目录
        self.query_dir = osp.join(self.data_dir, 'query')  # 查询集目录
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')  # 库集目录
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')  # 额外的库集目录（如果有 500k 图像）

        # 处理训练集、查询集和库集
        train = self.process_dir(self.train_dir, relabel=True)  # 处理训练集，重新标记 ID
        query = self.process_dir(self.query_dir, relabel=False)  # 处理查询集，不重新标记 ID
        gallery = self.process_dir(self.gallery_dir, relabel=False)  # 处理库集，不重新标记 ID
        if self.market1501_500k:  # 如果使用 500k 图像，则追加处理
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        # 调用父类的初始化方法
        super(Market1501, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        # 处理数据集目录中的图片，返回包含图片路径、行人 ID 和摄像头 ID 的列表
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))  # 获取目录下所有图片路径
        pattern = re.compile(r'([-\d]+)_c(\d)')  # 定义用于提取行人 ID 和摄像头 ID 的正则表达式

        pid_container = set()  # 创建集合存储所有行人 ID
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())  # 提取行人 ID 和摄像头 ID
            if pid == -1:
                continue  # 忽略垃圾图片（行人 ID 为 -1）
            pid_container.add(pid)  # 将有效的行人 ID 添加到集合中
        pid2label = {pid: label for label, pid in enumerate(pid_container)}  # 为每个行人 ID 分配新的标签

        data = []  # 创建用于存储图像数据的列表
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())  # 提取行人 ID 和摄像头 ID
            if pid == -1:
                continue  # 忽略垃圾图片
            assert 0 <= pid <= 1501  # 检查行人 ID 合法性（0 表示背景）
            assert 1 <= camid <= 6  # 确保摄像头 ID 合法（1-6）
            camid -= 1  # 将摄像头 ID 转为从 0 开始计数
            if relabel:
                pid = pid2label[pid]  # 如果需要重新标记 ID，则使用新标签替换原 ID
            data.append((img_path, pid, camid, 0))  # 将图像路径、行人 ID 和摄像头 ID 添加到列表中

        return data  # 返回处理后的数据

