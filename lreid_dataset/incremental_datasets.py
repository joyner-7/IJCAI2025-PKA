import numpy as np
from PIL import Image
import copy
import os
import copy
from prettytable import PrettyTable  # 引入prettytable库用于展示表格数据
from easydict import EasyDict
import random
from collections import defaultdict, OrderedDict
import operator


# 定义一个用于遍历目录的函数
def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):  # 遍历文件夹
        files = sorted(files, reverse=True)  # 按逆序排列文件
        dirs = sorted(dirs, reverse=True)  # 按逆序排列子目录
        return root, dirs, files  # 返回当前目录、子目录和文件列表


# 增量行人重识别样本类
class IncrementalPersonReIDSamples:

    # 重新给标签排序的函数
    def _relabels_incremental(self, samples, label_index, is_mix=False):
        '''
        重新排序标签，将标签 [1, 3, 5, 7] 映射到 [0,1,2,3]
        '''
        ids = [] # 初始化一个空列表，用于存储样本中的标签（身份ID）
        pid2label = {} # 初始化一个空字典，用于将每个身份ID映射到一个新的标签
        for sample in samples:
            ids.append(sample[label_index])  # 将样本的标签存入ids列表
        # 删除重复元素并排序
        ids = list(set(ids))  # 转换为集合去重
        ids.sort()  # 排序标签

        # 重新排序
        for sample in samples:
            sample = list(sample)  # 将样本转换为列表
            pid2label[sample[label_index]] = ids.index(sample[label_index])  # 标签重新映射

        # 深拷贝样本，防止修改原数据
        new_samples = copy.deepcopy(samples)
        for i, sample in enumerate(samples):
            new_samples[i] = list(new_samples[i])  # 转换为列表
            new_samples[i][label_index] = pid2label[sample[label_index]]  # 更新标签
        if is_mix:  # 如果是混合模式，返回原样本和映射后的标签
            return samples, pid2label
        else:  # 否则返回新的样本
            return new_samples

    # 加载图像路径的函数
    def _load_images_path(self, folder_dir, domain_name='market', is_mix=False):
        '''
        :param folder_dir: 文件夹路径
        :return: [(路径, 身份ID, 摄像头ID)]
        '''
        samples = []
        root_path, _, files_name = os_walk(folder_dir)  # 获取根目录、子目录和文件名
        for file_name in files_name:
            if '.jpg' in file_name:  # 检查文件是否为图片
                identi_id, camera_id = self._analysis_file_name(file_name, is_mix=is_mix)  # 解析文件名得到身份ID和摄像头ID
                samples.append([root_path + file_name, identi_id, camera_id, domain_name])  # 将路径、ID和域名添加到样本
        return samples  # 返回样本列表

    # 获取图像目录的属性函数
    @property
    def images_dir(self):
        return None

    # 解析文件名的函数
    def _analysis_file_name(self, file_name, is_mix=False):
        '''
        :param file_name: 格式类似 0844_c3s2_107328_01.jpg
        :return: 0844, 3 （身份ID，摄像头ID）
        '''

        # 将文件名中的.jpg移除，并替换字符，分割成列表
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        if is_mix:  # 如果是混合模式，身份ID和摄像头ID分别取第一个和第三个元素
            identi_id, camera_id = int(split_list[0]), int(split_list[2])
        else:  # 否则身份ID和摄像头ID分别取第一个和第二个元素
            identi_id, camera_id = int(split_list[0]), int(split_list[1])

        return identi_id, camera_id  # 返回身份ID和摄像头ID

    # 展示数据信息的函数
    def _show_info(self, train, query, gallery, name=None, if_show=True):
        if if_show:  # 如果需要展示信息
            def analyze(samples):
                pid_num = len(set([sample[1] for sample in samples]))  # 统计身份ID的数量
                cid_num = len(set([sample[2] for sample in samples]))  # 统计摄像头ID的数量
                sample_num = len(samples)  # 统计样本数量
                return sample_num, pid_num, cid_num  # 返回样本数、身份ID数和摄像头ID数

            try:
                train_info = analyze(train)  # 分析训练集信息
                query_info = analyze(query)  # 分析查询集信息
                gallery_info = analyze(gallery)  # 分析画廊集信息

                # 使用PrettyTable库展示数据集信息，安装方式：```pip install prettytable```
                table = PrettyTable(['set', 'images', 'identities', 'cameras'])  # 创建表格对象
                self.num_train_pids = train_info[1]  # 记录训练集身份ID数量
                table.add_row([self.__class__.__name__ if name is None else name, '', '', ''])  # 添加表格标题行
                table.add_row(['train', str(train_info[0]), str(train_info[1]), str(train_info[2])])  # 添加训练集信息
                table.add_row(['query', str(query_info[0]), str(query_info[1]), str(query_info[2])])  # 添加查询集信息
                table.add_row(['gallery', str(gallery_info[0]), str(gallery_info[1]), str(gallery_info[2])])  # 添加画廊集信息
                print(table)  # 输出表格
            except:  # 异常处理
                self.num_train_pids = 0  # 如果出错，将训练集身份ID数设为0
        else:
            pass  # 如果不展示信息，跳过





def Incremental_combine_test_samples(samples_list):
    '''combine more than one samples (e.g. market.train and duke.train) as a samples'''

    all_gallery, all_query = [], []

    def _generate_relabel_dict(s_list):
        pids_in_list, pid2relabel_dict = [], {}
        for new_label, samples in enumerate(s_list):
            if str(samples[1]) + str(samples[3]) not in pids_in_list:
                pids_in_list.append(str(samples[1]) + str(samples[3]))
        for i, pid in enumerate(sorted(pids_in_list)):
            pid2relabel_dict[pid] = i
        return pid2relabel_dict
    def _replace_pid2relabel(s_list, pid2relabel_dict, pid_dimension=1):
        new_list = copy.deepcopy(s_list)
        for i, sample in enumerate(s_list):
            new_list[i] = list(new_list[i])
            new_list[i][pid_dimension] = pid2relabel_dict[str(sample[pid_dimension])+str(sample[pid_dimension + 2])]
        return new_list

    for samples_class in samples_list:
        all_gallery.extend(samples_class.gallery)
        all_query.extend(samples_class.query)
    pid2relabel_dict = _generate_relabel_dict(all_gallery)
    # pid2relabel_dict2 = _generate_relabel_dict(all_query)

    # assert len(list(pid2relabel_dict2.keys())) == sum([1 for query_key in pid2relabel_dict2.keys() if query_key in pid2relabel_dict.keys()])
    #print(pid2relabel_dict)
    #print(pid2relabel_dict2)
    # assert operator.eq(pid2relabel_dict, _generate_relabel_dict(all_query))
    gallery = _replace_pid2relabel(all_gallery, pid2relabel_dict, pid_dimension=1)
    query = _replace_pid2relabel(all_query, pid2relabel_dict, pid_dimension=1)


    return query, gallery

def Incremental_combine_train_samples(samples_list):
    '''combine more than one samples (e.g. market.train and duke.train) as a samples'''
    all_samples, new_samples = [], []
    all_pid_per_step, all_cid_per_step, output_all_per_step = OrderedDict(), OrderedDict(), defaultdict(dict)
    max_pid, max_cid = 0, 0
    for step, samples in enumerate(samples_list):
        for a_sample in samples:
            img_path = a_sample[0]
            local_pid = a_sample[1]
            try:
                dataset_name = a_sample[3]
                global_pid = max_pid + a_sample[1]
                # global_cid = str(dataset_name) + ':' + str(a_sample[2])
                global_cid = max_cid + int(a_sample[2])
            except:
                print(a_sample)
                assert False
            all_samples.append([img_path, global_pid, global_cid, dataset_name, local_pid])
            if step in all_pid_per_step.keys():
                all_pid_per_step[step].add(global_pid)
            else:
                all_pid_per_step[step] = set()
                all_pid_per_step[step].add(global_pid)

            if step in all_cid_per_step.keys():
                all_cid_per_step[step].add(global_cid)
            else:
                all_cid_per_step[step] = set()
                all_cid_per_step[step].add(global_cid)
        for k, v in all_cid_per_step.items():
            all_cid_per_step[k] = sorted(v)
        max_pid = sum([len(v) for v in all_pid_per_step.values()])
        max_cid = sum([len(v) for v in all_cid_per_step.values()])

    return all_samples, all_pid_per_step, all_cid_per_step

class IncrementalReIDDataSet:
    def __init__(self, samples, total_step, transform):
        # 初始化函数，接收样本列表 samples，总步骤数 total_step 和图像变换函数 transform
        self.samples = samples  # 保存传入的样本列表
        self.transform = transform  # 保存图像变换函数
        self.total_step = total_step  # 保存总步骤数

    def __getitem__(self, index):
        # 定义获取指定索引的样本的方式，索引通过 index 参数传入

        this_sample = copy.deepcopy(self.samples[index])
        # 深拷贝样本，防止原始样本数据被修改
        this_sample = list(this_sample)
        # 将样本转换为列表格式
        this_sample.append(this_sample[0])
        # 将样本的第一个元素（通常是图像路径）追加到列表末尾，方便后续处理
        this_sample[0] = self._loader(this_sample[0])
        # 使用 _loader 函数加载图像
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
            # 如果存在图像变换函数，则对加载后的图像进行变换
        this_sample[1] = np.array(this_sample[1])
        # 将样本的第二个元素（通常是标签）转换为 NumPy 数组

        return this_sample
        # 返回处理后的样本

    def __len__(self):
        # 返回样本集的长度
        return len(self.samples)

    def _loader(self, img_path):
        # 定义图像加载器，接收图像路径 img_path，打开图像并转换为 RGB 格式
        return Image.open(img_path).convert('RGB')

