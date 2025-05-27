from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
import torch

from .evaluation_metrics import cmc, mean_ap, mean_ap_cuhk03
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from torch.nn import functional as F

def extract_features(model, data_loader,training_phase=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domians) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model,imgs)#,training_phase=training_phase
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

    return features, labels

def extract_features_print(model, data_loader,training_phase=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domians) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs,training_phase=training_phase)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

    return features, labels




def pairwise_distance(features, query=None, gallery=None, metric=False):  # 定义一个函数 pairwise_distance，接受 features、query、gallery 和 metric 作为参数
    if query is None and gallery is None:  # 如果 query 和 gallery 都为 None，执行以下操作
        n = len(features)  # 获取 features 的长度 n
        x = torch.cat(list(features.values()))  # 将 features 中所有的值拼接成一个张量 x
        x = x.view(n, -1)  # 将 x 重新调整形状，变为 n 行
        if metric is not False:  # 如果 metric 不为 False，进行归一化
            x = F.normalize(x, p=2, dim=1)  # 对 x 进行 L2 归一化
            # x = metric.transform(x)  # 这里注释掉了 metric 的变换操作
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2  # 计算 x 的平方和，保留维度，然后乘以 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())  # 扩展 dist_m 维度并进行矩阵乘法，计算对角线距离
        return dist_m  # 返回距离矩阵 dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _,_ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _,_ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not False:
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        # x = metric.transform(x)
        # y = metric.transform(y)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m.cpu(), x.cpu().numpy(), y.cpu().numpy()

def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False, cuhk03=False):
    # 如果 query 和 gallery 不为空，则从中提取 query_ids 和 gallery_ids 以及 query_cams 和 gallery_cams
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _,_ in query]  # 从 query 中提取每个样本的 pid
        gallery_ids = [pid for _, pid, _,_ in gallery]  # 从 gallery 中提取每个样本的 pid
        query_cams = [cam for _, _, cam,_ in query]  # 从 query 中提取每个样本的摄像头编号 cam
        gallery_cams = [cam for _, _, cam,_ in gallery]  # 从 gallery 中提取每个样本的摄像头编号 cam
    else:
        # 否则，需要确保 query_ids、gallery_ids、query_cams 和 gallery_cams 都已提供
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # 计算平均精度均值（Mean AP）
    if cuhk03:
        # 如果是 cuhk03 数据集，调用 mean_ap_cuhk03 函数计算 mAP
        mAP = mean_ap_cuhk03(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    else:
        # 否则，调用 mean_ap 函数计算 mAP
        mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))  # 打印 mAP

    # 如果不需要计算 CMC，则直接返回 mAP
    if (not cmc_flag):
        return mAP

    '''
    下面注释掉的部分包含 market1501 和 cuhk03 两种配置下的 CMC 计算方法，
    可以根据具体的评估需要解注部分代码
    '''
    '''
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),
        'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False)
                }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
    '''

    # 如果是 cuhk03 数据集，设置对应的 CMC 配置
    if cuhk03:
        cmc_configs = {
        'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False)
                }
        
        # 根据 cmc_configs 计算 cmc 分数
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
        print('CUHK03 CMC Scores:')  # 打印 cuhk03 的 CMC 分数
        # 遍历并打印 CMC top-k 结果
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'
                .format(k,
                        cmc_scores['cuhk03'][k-1]))
        return cmc_scores['cuhk03'][0], mAP  # 返回 top-1 CMC 和 mAP
    
    else:
        # 如果是 market1501 数据集，设置对应的 CMC 配置
        cmc_configs = {
            'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),
                }
        # 根据 cmc_configs 计算 cmc 分数
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
        print('CMC Scores:')  # 打印 market1501 的 CMC 分数
        # 遍历并打印 CMC top-k 结果
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'
                .format(k,
                        cmc_scores['market1501'][k-1]))
        return cmc_scores['market1501'][0], mAP  # 返回 top-1 CMC 和 mAP

class Evaluator(object):
    # 初始化函数，传入模型并将其赋值给类成员变量 self.model
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    # 评估函数，接收数据加载器 data_loader、query、gallery 等参数，执行评估过程
    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False,
                 rerank=False, pre_features=None, cuhk03=False, training_phase=None):
        # 如果 pre_features 为空，则从模型中提取特征
        if (pre_features is None):
            # 提取特征，调用 extract_features 函数
            features, _ = extract_features(self.model, data_loader)  #,training_phase=training_phase 可选参数
        else:
            # 如果 pre_features 不为空，则直接使用已有的特征
            features = pre_features

        # 计算特征之间的距离矩阵，调用 pairwise_distance 函数计算 distmat，
        # query_features 和 gallery_features 分别表示查询和图库的特征
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)

        # 调用 evaluate_all 函数进行评估，返回评估结果
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag, cuhk03=cuhk03)

        # 如果不需要 rerank（重排序），则直接返回评估结果
        if (not rerank):
            return results

        # 如果需要进行 rerank，则打印相关信息并进行重新排序操作
        print('Applying person re-ranking ...')

        # 计算 query 到 query，gallery 到 gallery 的距离矩阵
        distmat_qq = pairwise_distance(features, query, query, metric=metric)
        distmat_gg = pairwise_distance(features, gallery, gallery, metric=metric)

        # 调用 re_ranking 函数对距离矩阵进行重新排序
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())

        # 重新评估排序后的距离矩阵，调用 evaluate_all 函数，并返回结果
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

