import torch
import torch.nn.functional as F

from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from .data.sampler import RandomIdentitySampler, MultiDomainRandomIdentitySampler

import collections
import numpy as np
import copy

def extract_features_adv(model, data_loader):
    features_all = []
    labels_all = []
    model.eval()
    with torch.no_grad():
        for i, (imgs,pids) in enumerate(data_loader):
            features = model(imgs)
            for feature, pid in zip(features, pids):
                features_all.append(feature)
                labels_all.append(int(pid))
    model.train()
    return features_all, labels_all

def extract_features(model, data_loader):
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    model.eval()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domains) in enumerate(data_loader):
            #print(f"Processing batch {i + 1}: {len(imgs)} images")  # 打印当前批次的信息
            features= model(imgs)[0]
            for fname, feature, pid, cid in zip(fnames, features, pids, cids):
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)
    #print('finished extract_features')
    model.train()
    return features_all, labels_all, fnames_all, camids_all

def extract_features_iter(model, data_loader):
    #print('extract_features_iter')
    features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    model.eval()
    with torch.no_grad():
        for i in range(len(data_loader)):  # 可能需要用 __len__ 来获取总批次数
            imgs, fnames, pids, cids, domains = data_loader.next()  # 使用 next 获取数据
            features= model(imgs)
            for fname, feature, pid, cid in zip(fnames, features, pids, cids):
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)
    #print('finished extract_features')
    model.train()
    return features_all, labels_all, fnames_all, camids_all


def initial_classifier(model, data_loader):
    pid2features = collections.defaultdict(list)
    features_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader)
    for feature, pid in zip(features_all, labels_all):
        pid2features[pid].append(feature)
    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = torch.stack(class_centers)
    return F.normalize(class_centers, dim=1).float().cuda()

def obtain_voronoi_loader(dataset,new_labels, add_num=0, batch_size = 32,num_instance=4,workers=8):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    voronoi_set=[]

    if not isinstance(new_labels, list):
        new_labels = new_labels.tolist()

    for instance, lablel in zip(dataset.train, new_labels):    
        a=(instance[0], lablel, instance[2], instance[3])
        voronoi_set.append(a

        )     

    voronoi_loader = DataLoader(Preprocessor(voronoi_set, dataset.images_dir, train_transformer),
                             batch_size=batch_size,num_workers=workers, sampler=RandomIdentitySampler(voronoi_set, num_instance),
                             pin_memory=True, drop_last=True)
    # new_labels.sort()

    return voronoi_loader, voronoi_set
def extract_features_uncertain(model, data_loader, get_mean_feature=True):
    features_all = []  # 保存所有特征的列表
    labels_all = []  # 保存所有标签（人物ID）的列表
    fnames_all = []  # 保存所有文件名的列表
    camids_all = []  # 保存所有摄像头ID的列表
    var_all = []  # 保存所有特征不确定性（方差）的列表
    model.train()  # 将模型设置为训练模式，以启用特定层的行为
    with torch.no_grad():  # 关闭梯度计算，因为这是特征提取，不需要反向传播
        for i, (imgs, fnames, pids, cids, domains) in enumerate(data_loader):
            #print('type(data_loader) =',type(data_loader))
            # 从模型中获取输出，包括平均特征、合并特征、分类输出、不确定性和其他信息
            mean_feat, merge_feat, cls_outputs, out_var, _ = model(imgs)
            # 将每个文件的特征、标签、文件名、摄像头ID和不确定性添加到各自的列表中
            for fname, feature, pid, cid, var in zip(fnames, mean_feat, pids, cids, out_var):
                features_all.append(feature)
                labels_all.append(int(pid))
                fnames_all.append(fname)
                camids_all.append(cid)
                var_all.append(var)

    # 如果不需要计算均值特征，直接返回收集到的特征、标签、文件名和摄像头ID
    if get_mean_feature:
        features_collect = {}  # 用于存储每个标签对应的特征列表
        var_collect = {}  # 用于存储每个标签对应的不确定性列表

        # 按标签收集每个人物的特征和不确定性
        for feature, label, var in zip(features_all, labels_all, var_all):
            if label in features_collect:
                features_collect[label].append(feature)
                var_collect[label].append(var)
            else:
                features_collect[label] = [feature]
                var_collect[label] = [var]
                
        labels_named = list(set(labels_all))  # 去除重复的标签，获得有效的标签列表
        labels_named.sort()  #对标签进行排序
        features_mean = []  #保存均值特征的列表
        vars_mean = []  #保存均值不确定性的列表
        
        # 计算每个标签的均值特征和均值不确定性
        for x in labels_named:
            if x in features_collect.keys():
                feats = torch.stack(features_collect[x])  # 将特征列表堆叠成一个张量
                feat_mean = feats.mean(dim=0)  # 计算特征的均值
                features_mean.append(feat_mean)

                # 计算不确定性的均值，公式中使用方差的均值
                #vars_2 = (torch.stack(var_collect[x])**2).mean(dim=0) + (feats**2).mean(dim=0) - feat_mean**2
                #vars_mean.append(torch.sqrt(vars_2))  # 计算标准差并添加到列表
            else:
                features_mean.append(torch.zeros_like(features_all[0]))  # 若无特征，则返回与特征维度相同的全零张量
                #vars_mean.append(torch.zeros_like(var_all[0]))  # 若无不确定性，则返回全零张量
        #返回所有特征、标签、文件名、摄像头ID、均值特征、有效标签和均值不确定性
        return features_all, labels_all, fnames_all, camids_all, torch.stack(features_mean), labels_named#, torch.stack(vars_mean), var_all
    else:
        # 如果不需要均值特征，直接返回所有特征、标签、文件名和摄像头ID
        return features_all, labels_all, fnames_all, camids_all

def extract_features_voro(model, data_loader, get_mean_feature=False):    

    features_all, labels_all, fnames_all, camids_all = extract_features(model, data_loader)

    if get_mean_feature:
        features_collect = {}        

        for feature, label in zip(features_all, labels_all):
            if label in features_collect:
                features_collect[label].append(feature)               
            else:
                features_collect[label] = [feature]                
        labels_named = list(set(labels_all))  # obtain valid features
        labels_named.sort()
        features_mean=[]        
        for x in labels_named:
            if x in features_collect.keys():
                features_mean.append(torch.stack(features_collect[x]).mean(dim=0))                
            else:
                features_mean.append(torch.zeros_like(features_all[0]))
                
        return features_all, labels_all, fnames_all, camids_all, torch.stack(features_mean),labels_named
    else:
        return features_all, labels_all, fnames_all, camids_all



def select_replay_samples(model, dataset, training_phase=0, add_num=0, old_datas=None, select_samples=2,batch_size = 32,workers=8):
    replay_data = []
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_loader = DataLoader(Preprocessor(dataset.train, root=dataset.images_dir,transform=transformer),
                              batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True, drop_last=False)

    features_all, labels_all, fnames_all, camids_all = extract_features(model, train_loader)

    pid2features = collections.defaultdict(list)
    pid2fnames = collections.defaultdict(list)
    pid2cids = collections.defaultdict(list)

    for feature, pid, fname, cid in zip(features_all, labels_all, fnames_all, camids_all):
        pid2features[pid].append(feature)
        pid2fnames[pid].append(fname)
        pid2cids[pid].append(cid)

    labels_all = list(set(labels_all))

    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = F.normalize(torch.stack(class_centers), dim=1)
    select_pids = np.random.choice(labels_all, 250, replace=False)
    for pid in select_pids:
        feautures_single_pid = F.normalize(torch.stack(pid2features[pid]), dim=1, p=2)
        center_single_pid = class_centers[pid]
        simi = torch.mm(feautures_single_pid, center_single_pid.unsqueeze(0).t())
        simi_sort_inx = torch.sort(simi, dim=0)[1][:2]
        for id in simi_sort_inx:
            replay_data.append((pid2fnames[pid][id], pid+add_num, pid2cids[pid][id], training_phase-1))

    if old_datas is None:
        data_loader_replay = DataLoader(Preprocessor(replay_data, dataset.images_dir, train_transformer),
                             batch_size=batch_size,num_workers=workers, sampler=RandomIdentitySampler(replay_data, select_samples),
                             pin_memory=True, drop_last=True)
    else:
        replay_data.extend(old_datas)
        data_loader_replay = DataLoader(Preprocessor(replay_data, dataset.images_dir, train_transformer),
                             batch_size=training_phase*batch_size,num_workers=workers,
                             sampler=MultiDomainRandomIdentitySampler(replay_data, select_samples),
                             pin_memory=True, drop_last=True)

    return data_loader_replay, replay_data


def get_pseudo_features(data_specific_batch_norm, training_phase, x, domain, unchange=False):
    fake_feat_list = []
    if unchange is False:
        for i in range(training_phase):
            if int(domain[0]) == i:
                data_specific_batch_norm[i].train()
                fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])
            else:
                data_specific_batch_norm[i].eval()
                fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])
                data_specific_batch_norm[i].train()
    else:
        for i in range(training_phase):
            data_specific_batch_norm[i].eval()
            fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])

    return fake_feat_list
