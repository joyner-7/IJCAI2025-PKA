from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import sys

import pdb

import matplotlib
matplotlib.use('Agg')  # 使用非图形界面的后端
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
from torch.backends import cudnn
import torch.nn as nn
import random
from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from reid.models.resnet_uncertainty import ResNetSimCLR
from reid.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from tools.Logger_results import Logger_res
def main():
    args = parser.parse_args()

    if args.seed is not None:
        print("setting the seed to",args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    log_name = 'log.txt'
    if not args.evaluate:#训练模式
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:#测试模式
        log_dir = osp.dirname(args.test_folder)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))
    log_res_name='log_res.txt'
    logger_res=Logger_res(osp.join(args.logs_dir, log_res_name))    # record the test results
    

    """
    loading the datasets:
    setting： 1 or 2 
    """
    if 1 == args.setting:
        training_set = ['market1501','cuhk_sysu','dukemtmc','msmt17','cuhk03']
        #,'cuhk_sysu','cuhk03'
    else:
        training_set = ['market1501','dukemtmc', 'msmt17']
        #, 'cuhk_sysu', 'cuhk03'
        #
    # all the revelent datasets
        
    all_set = ['market1501','sense','prid', 'dukemtmc', 'msmt17',
    'cuhk01','cuhk02']  #        
    # the datsets only used for testing 'grid', 'viper','cuhk03','cuhk_sysu',
    testing_only_set = [x for x in all_set if x not in training_set]
     
    # get the loders of different datasets
    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)    
    
    first_train_set = all_train_sets[0]
    print('len training_set',len(training_set))
    print('len testing_only_set',len(testing_only_set))

    # 使用实际映射后的设备编号
    device_ids = [0,1]  # 逻辑设备 ID 对应物理 GPU 4, 5, 6, 7

    # 伪映射逻辑：将逻辑设备映射到物理设备
    logical_device_map = {0:0,1:1}  # 映射关系
    main_device = torch.device(f'cuda:{logical_device_map[device_ids[0]]}')  # 主 GPU 对应物理 4

    main_device = torch.device(f'cuda:{device_ids[0]}')  # 主 GPU 设置为 4 号显卡
    model=ResNetSimCLR(num_classes=first_train_set[1], uncertainty=True,n_sampling=args.n_sampling)
    model.to(main_device)
    
    #model.cuda() #将模型移动到 GPU 上
    model = DataParallel(model, device_ids=device_ids)
    #model = DataParallel(model)
    #torch.cuda.set_device(device_ids[0])
    writer = SummaryWriter(log_dir=args.logs_dir)
    # Load from checkpoint
    '''test the models under a folder'''
    if args.test_folder:
        ckpt_name = [x + '_checkpoint.pth.tar' for x in training_set]   # obatin pretrained model name
        checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[0]))  # load the first model
        copy_state_dict(checkpoint['state_dict'], model)     #    
        for step in range(len(ckpt_name) - 1):
            model_old = copy.deepcopy(model)    # backup the old model            
            checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[step + 1]))
            copy_state_dict(checkpoint['state_dict'], model)                         
           
            model = linear_combination(args, model, model_old, 0.5)

            save_name = '{}_checkpoint_adaptive_ema.pth.tar'.format(training_set[step+1])
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0,
                'mAP': 0,
            }, True, fpath=osp.join(args.logs_dir, save_name))        
        test_model(model, all_train_sets, all_test_only_sets, len(all_train_sets)-1,logger_res=logger_res)

        exit(0)
    

    # resume from a model
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
   
    # Evaluator
    if args.MODEL in ['50x']:
        out_channel = 2048
    else:
        raise AssertionError(f"the model {args.MODEL} is not supported!")
    
    proto_type={}
    # train on the datasets squentially
    for set_index in range(0, len(training_set)):       
        model_old = copy.deepcopy(model)
        model= train_dataset(args, proto_type,all_train_sets, all_test_only_sets, set_index, model, out_channel,
                                            writer,logger_res=logger_res)
        #test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res)

        if set_index>0:
            add_num = sum([all_train_sets[i][1] for i in range(set_index)])  # get model out_dim
            dataset, num_classes, train_loader, test_loader, init_loader, name,pic_num = all_train_sets[set_index]  # status of current dataset

            pic_num_total = sum([all_train_sets[i][6] for i in range(set_index)])  # get model out_dim
            print(f"pic_num: {pic_num}")
            print(f"pic_num_total: {pic_num_total}")
            alpha = pic_num / pic_num_total  # calculate the alpha for linear combination
            model=linear_combination(args, model, model_old, alpha)  
            #test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res)    
    print('finished')


def obtain_old_types( all_train_sets, set_index, model):
    dataset_old, num_classes_old, train_loader_old, _, init_loader_old, name_old,pic_num = all_train_sets[set_index - 1]  # trainloader of old dataset
    features_all_old, labels_all_old, fnames_all, camids_all, features_mean, labels_named= extract_features_uncertain(model,
                                                                                                                  init_loader_old,
                                                                                                                  get_mean_feature=True)  # init_loader is original designed for classifer init
    #, vars_mean,vars_all 
    features_all_old = torch.stack(features_all_old)
    labels_all_old = torch.tensor(labels_all_old).to(features_all_old.device)
    features_all_old.requires_grad = False
    return features_all_old, labels_all_old, features_mean, labels_named#,vars_mean,vars_all


def train_dataset(args, proto_type,all_train_sets, all_test_only_sets, set_index, model, out_channel, writer,logger_res=None):
    if set_index>0:
        features_all_old, labels_all_old,features_mean, labels_named=obtain_old_types( all_train_sets,set_index,model)
        proto_type={}
        #,vars_mean,vars_all
        proto_type[set_index-1]={
                "features_all_old":features_all_old,
                "labels_all_old":labels_all_old,
                'mean_features':features_mean,
                'labels':labels_named,
                #'mean_vars':vars_mean,
                #"vars_all":vars_all
        }
    else:
        proto_type=None
        dataset, num_classes, train_loader, test_loader, init_loader, name,picnum= all_train_sets[set_index]  # status of current dataset    

    Epochs= args.epochs0 if 0==set_index else args.epochs
    dataset, num_classes, train_loader, test_loader, init_loader, name,picnum= all_train_sets[set_index]  # status of current dataset

    Epochs= args.epochs0 if 0==set_index else args.epochs          

    if set_index<1:
        add_num = 0
        old_model=None
    else:
        add_num = sum(
            [all_train_sets[i][1] for i in range(set_index - 1)])  # get person number in existing domains
        #print('add_num',add_num)
    
    if set_index>0:
        '''store the old model'''
        old_model = copy.deepcopy(model)
        old_model = old_model.cuda()
        old_model.eval()

        # after sampling rehearsal, recalculate the addnum(historical ID number)
        add_num = sum([all_train_sets[i][1] for i in range(set_index)])  # get model out_dim

        pic_num = sum([all_train_sets[i][6] for i in range(set_index)])
        print(f"add_num: {add_num}")
        print(f"pic_num: {pic_num}")
        print('num_classes =' , num_classes)

        # Expand the dimension of classifier
        org_classifier_params = model.module.classifier.weight.data

        # 打印 org_classifier_params 的 shape
        print(f"Original classifier params shape: {org_classifier_params.shape}")

        model.module.classifier = nn.Linear(out_channel, add_num + num_classes, bias=False)

        # 打印新的 classifier 的 shape
        print(f"New classifier weight shape: {model.module.classifier.weight.data.shape}")

        # 将原有的参数复制到新的 classifier
        model.module.classifier.weight.data[:add_num].copy_(org_classifier_params)
        model.cuda()

        # Initialize classifier with class centers
        class_centers = initial_classifier(model, init_loader)

        # 打印 class_centers 的 shape
        print(f"Class centers shape: {class_centers.shape}")

        # 打印 add_num 之后的 classifier weight 的 shape
        print(f"Classifier weight shape after add_num: {model.module.classifier.weight.data[add_num:].shape}")

        # 将 class_centers 的值复制到 classifier 对应大小的位置
        model.module.classifier.weight.data[add_num:add_num + class_centers.shape[0]].copy_(class_centers)

        model.cuda()


    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            print('not requires_grad:', key)
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=args.momentum)    
    # Stones=args.milestones
    Stones = [20, 30] if name == 'msmt17' else args.milestones
    lr_scheduler = WarmupMultiStepLR(optimizer, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    
  
    trainer = Trainer(args, model, old_model, writer=writer)

    print('####### starting training on {} #######'.format(name))
    for epoch in range(0, Epochs):#epoch
        train_loader.new_epoch()

        trainer.train(epoch, train_loader,  optimizer, training_phase=set_index + 1,
                      proto_type=proto_type,train_iters=len(train_loader), add_num=add_num
                      )
 
        lr_scheduler.step()       
       

        if ((epoch + 1) % args.eval_epoch == 0 or epoch+1==Epochs):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': 0.,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))

            logger_res.append('epoch: {}'.format(epoch + 1))
            
            mAP=0.
            if args.middle_test:
                mAP = test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res)                    
          
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))
  
    return model


def test_model(model, all_train_sets, all_test_sets, set_index,  logger_res=None):
    begin = 0
    evaluator = Evaluator(model)
        
    R1_all = []
    mAP_all = []
    names=''
    Results=''
    train_mAP=0
    for i in range(begin, set_index + 1):
        dataset, num_classes, train_loader, test_loader, init_loader, name,pic_num = all_train_sets[i]
        print('Results on {}'.format(name))
        train_R1, train_mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery,
                                                 cmc_flag=True)  # ,training_phase=i+1)
        R1_all.append(train_R1)
        mAP_all.append(train_mAP)
        names = names + name + '\t\t'
        Results=Results+'|{:.1f}/{:.1f}\t'.format(train_mAP* 100, train_R1* 100)

    aver_mAP = torch.tensor(mAP_all).mean()
    aver_R1 = torch.tensor(R1_all).mean()

    R1_all = []
    mAP_all = []
    names_unseen = ''
    Results_unseen = ''
    for i in range(len(all_test_sets)):
        dataset, num_classes, train_loader, test_loader, init_loader, name,pic_num = all_test_sets[i]
        print('Results on {}'.format(name))
        R1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery,
                                     cmc_flag=True)

        R1_all.append(R1)
        mAP_all.append(mAP)
        names_unseen = names_unseen + name + '\t'
        Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t'.format(mAP* 100, R1* 100)

    aver_mAP_unseen = torch.tensor(mAP_all).mean()
    aver_R1_unseen = torch.tensor(R1_all).mean()

    print("Average mAP on Seen dataset: {:.1f}%".format(aver_mAP * 100))
    print("Average R1 on Seen dataset: {:.1f}%".format(aver_R1 * 100))
    names = names + '|Average\t|'
    Results = Results + '|{:.1f}/{:.1f}\t|'.format(aver_mAP * 100, aver_R1 * 100)
    print(names)
    print(Results)
    '''_________________________'''
    print("Average mAP on unSeen dataset: {:.1f}%".format(aver_mAP_unseen * 100))
    print("Average R1 on unSeen dataset: {:.1f}%".format(aver_R1_unseen * 100))
    names_unseen = names_unseen + '|Average\t|'
    Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t|'.format(aver_mAP_unseen* 100, aver_R1_unseen* 100)
    print(names_unseen)
    print(Results_unseen)
    if logger_res:
        logger_res.append(names)
        logger_res.append(Results)
        logger_res.append(Results.replace('|','').replace('/','\t'))
        logger_res.append(names_unseen)
        logger_res.append(Results_unseen)
        logger_res.append(Results_unseen.replace('|', '').replace('/', '\t'))
    return train_mAP


def visualize_tsne_for_ids(all_train_sets, model, device, save_dir='/data1/lzj_log/COPY', num_ids=5):
    """
    从五个训练集中随机抽取 num_ids 个 ID，将这些 ID 的所有样本（来自训练集 train_loader） 
    都可视化在同一张 t-SNE 散点图上。
    相同 ID 用同一颜色表示，以观察聚类效果。
    
    Args:
        all_train_sets: 五个训练数据集的信息列表，每个元素包含 
            (dataset, num_classes, train_loader, test_loader, init_loader, name, pic_num)
        model: 已训练好的模型，用于提取特征
        device: CPU 或 GPU 设备
        save_dir: 保存可视化结果的文件夹
        num_ids: 随机选取的 ID 数量
    """
    import random
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import os.path as osp

    print("Starting t-SNE visualization for randomly selected IDs from train_loader ...")
    model.eval()

    # 用字典来存储所有 ID 的特征：key=ID, value=该 ID 所有样本的特征(list)
    all_features_dict = {}

    # 遍历所有训练域
    for domain_data in all_train_sets:
        dataset, num_classes, train_loader, test_loader, init_loader, name, pic_num = domain_data
        
        # 改为使用 train_loader 抽取特征
        # extract_features_uncertain 返回：features_list, labels_list, ...
        features_list, labels_list, _, _= extract_features_iter(
            model, 
            train_loader
        )

        # 将对应 ID 的特征放到字典里
        for feat, label in zip(features_list, labels_list):
            if label not in all_features_dict:
                all_features_dict[label] = []
            # 注意：feat 通常是 torch.Tensor，需要转到 CPU 再变成 numpy
            all_features_dict[label].append(feat.cpu().numpy())

    # 收集所有出现过的 ID
    all_ids = list(all_features_dict.keys())

    # 若总 ID 数不足 num_ids，就全拿；否则随机采样 num_ids 个
    if len(all_ids) <= num_ids:
        selected_ids = all_ids
    else:
        selected_ids = random.sample(all_ids, num_ids)
    print(f"Selected {len(selected_ids)} IDs from total {len(all_ids)} IDs.")

    # 准备 t-SNE 的输入特征和标签
    selected_features = []
    selected_labels = []
    for label_id in selected_ids:
        feats_for_id = all_features_dict[label_id]
        # 将该 ID 的全部样本加入可视化
        for f in feats_for_id:
            selected_features.append(f)
            selected_labels.append(label_id)

    # 转成 numpy array
    selected_features = np.array(selected_features)  # shape: [N, feat_dim]
    selected_labels = np.array(selected_labels)

    print(f"Total shape of features for selected IDs: {selected_features.shape}")

    # 执行 t-SNE 降维
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    features_2d = tsne.fit_transform(selected_features)  # [N, 2]

    # 开始绘图
    plt.figure(figsize=(14, 12))
    unique_ids = np.unique(selected_labels)
    # 常见调色板：tab10 或 tab20, ID 超过 10/20 则需自定义颜色映射
    cmap = plt.cm.get_cmap('tab10', len(unique_ids))

    for i, uid in enumerate(unique_ids):
        # 取出当前 ID 的所有样本点
        indices = (selected_labels == uid)
        plt.scatter(
            features_2d[indices, 0],
            features_2d[indices, 1],
            label=f'ID {uid}',
            s=10,       # 点的大小
            alpha=0.7,  # 透明度
            color=cmap(i)
        )

    # 去除坐标轴边框和刻度线做美观处理
    ax = plt.gca()
    ax.spines['top'].set_visible(False)    
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='best', frameon=False, fontsize=8)  # 图例可根据需要调整

    # 保存结果
    save_path = osp.join(save_dir, 'tsne_random_ids.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"t-SNE visualization saved to {save_path}")


def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    '''old model '''
    model_old_state_dict = model_old.state_dict()
    '''latest trained model'''
    model_state_dict = model.state_dict()
    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    '''fuse the parameters'''
    for k, v in model_state_dict.items():
        if model_old_state_dict[k].shape == v.shape:
            # print(k,'+++')
                model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            print(k, '...')
            num_class_old = model_old_state_dict[k].shape[0]
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k]
    model_new.load_state_dict(model_new_state_dict)
    return model_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model    
    parser.add_argument('--MODEL', type=str, default='50x',
                        choices=['50x'])
    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help="optimizer ")
    parser.add_argument('--lr', type=float, default=0.008,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=200)

    # path   
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/data0/data_lzj/')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('/data1/lzj_log/COPY'))

      
    parser.add_argument('--test_folder', type=str, default='', help="test the models in a file")
   
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2], help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--AF_weight', default=0.1, type=float, help="anti-forgetting weight")   
    parser.add_argument('--n_sampling', default=6, type=int, help="number of sampling by Gaussian distribution")
    parser.add_argument('--lambda_1', default=0.1, type=float, help="temperature")
    parser.add_argument('--lambda_2', default=0.1, type=float, help="temperature")
    parser.add_argument('--score_batch_percentage', default=0.25, type=float, help="score_batch_percentage")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                    type=str, choices=['cuda', 'cpu'], help="Training device")   
    main()
