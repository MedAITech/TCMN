import datetime
import os.path
from os import path
import math

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed

from model.model import STCNModel
from dataset.static_dataset import StaticTransformDataset
from dataset.vos_dataset import VOSDataset
from dataset.med_dataset import MEDDataset

from util.logger import TensorboardLogger
from util.hyper_para import HyperParameters
from util.load_subset import load_sub_davis, load_sub_yv
import tqdm

"""
Initial setup
"""
# Init distributed environment
# distributed.init_process_group(backend="nccl")
# Set seed to ensure the same initialization
seed = 14159265
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

print('CUDA Device count: ', torch.cuda.device_count())

# Parse command line arguments
para = HyperParameters()
para.parse()

if para['benchmark']:
    torch.backends.cudnn.benchmark = True

# local_rank = torch.distributed.get_rank()
# world_size = torch.distributed.get_world_size()
# torch.cuda.set_device(local_rank)
world_size = 1
local_rank = para['local_rank']
torch.cuda.set_device(local_rank)

# print('I am rank %d in this world of size %d!' % (local_rank, world_size))

"""
Model related
"""
# if local_rank == 0:
# Logging
if para['id'].lower() != 'null':
    print('I will take the role of logging!')
    long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), para['id'])
else:
    long_id = None
logger = TensorboardLogger(para['id'], long_id)
logger.log_string('hyperpara', str(para))

# Construct the rank 0 model
model = STCNModel(para, logger=logger,
                  save_path=path.join('saves', long_id, long_id) if long_id is not None else None,
                  local_rank=local_rank, world_size=world_size).train()
# else:
#     # Construct model for other ranks
#     model = STCNModel(para, local_rank=local_rank, world_size=world_size).train()

# Load pertrained model if needed
if para['load_model'] is not None:
    total_iter = model.load_model(para['load_model'])
    print('Previously trained model loaded!')
else:
    total_iter = 0

if para['load_network'] is not None:
    model.load_network(para['load_network'])
    print('Previously trained network loaded!')

"""
Dataloader related
"""


# To re-seed the randomness everytime we start a worker
def worker_init_fn(worker_id):
    return np.random.seed(torch.initial_seed() % (2 ** 31) + worker_id + local_rank * 100)


def construct_loader(dataset):
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
    train_sampler = torch.utils.data.RandomSampler(dataset)
    # train_loader = DataLoader(dataset, para['batch_size'], sampler=train_sampler, num_workers=para['num_workers'],
    #                           worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True)
    train_loader = DataLoader(dataset, para['batch_size'], sampler=train_sampler, num_workers=para['num_workers'],
                              drop_last=True, pin_memory=True)
    return train_sampler, train_loader


def renew_vos_loader(max_skip):
    # //5 because we only have annotation for every five frames
    # yv_dataset = VOSDataset(path.join(ASU_root, 'video'),
    #                         path.join(ASU_root, 'mask'), max_skip // 5, is_bl=False, subset=load_sub_yv())
    # davis_dataset = VOSDataset(path.join(CVCClinic_root, 'JPEGImages', '480p'),
    #                            path.join(davis_root, 'Annotations', '480p'), max_skip, is_bl=False,
    #                            subset=load_sub_davis())
    # asu_dataset = MEDDataset(path.join(ASU_root, 'video'),
    #                          path.join(ASU_root, 'mask'), max_skip, is_bl=False)
    # cvc600_dataset = MEDDataset(path.join(CVCClinic_root, 'video'),
    #                             path.join(CVCClinic_root, 'mask'), max_skip // 5, is_bl=False)
    # cvc300_dataset = MEDDataset(path.join(CVCColon_root, 'video'),
    #                             path.join(CVCColon_root, 'mask'), max_skip // 5, is_bl=False)
    # hmc_dataset = MEDDataset(path.join(HMCQU_root, 'video'),
    #                          path.join(HMCQU_root, 'mask'), max_skip, is_bl=False)
    # camus_2ch_dataset = MEDDataset(path.join(CAMUS_2CH_root, 'video'),
    #                                path.join(CAMUS_2CH_root, 'mask'), max_skip, is_bl=False)
    # camus_4ch_dataset = MEDDataset(path.join(CAMUS_4CH_root, 'video'),
    #                                path.join(CAMUS_4CH_root, 'mask'), max_skip, is_bl=False)
    inhouse_dataset = MEDDataset(path.join(IN_HOUSE_root, 'video'),
                                   path.join(IN_HOUSE_root, 'mask'), max_skip, is_bl=False)

    # train_dataset = ConcatDataset([asu_dataset] + [cvc600_dataset] + [cvc300_dataset] + [hmc_dataset])
    # train_dataset = ConcatDataset([asu_dataset] + [cvc600_dataset] + [hmc_dataset])
    # train_dataset = ConcatDataset([camus_2ch_dataset] + [camus_4ch_dataset] + [hmc_dataset])
    train_dataset = ConcatDataset([inhouse_dataset])
    # train_dataset = ConcatDataset([cvc600_dataset] + [hmc_dataset])

    # print('ASU dataset size: ', len(asu_dataset))
    # print('CVC600 dataset size: ', len(cvc600_dataset))
    # print('CVC300 dataset size: ', len(cvc300_dataset))
    # print('HMC dataset size: ', len(hmc_dataset))
    # print('camus_4ch dataset size: ', len(camus_4ch_dataset))
    # print('camus_2ch dataset size: ', len(camus_2ch_dataset))
    print('inhouse dataset size: ', len(inhouse_dataset))
    print('Concat dataset size: ', len(train_dataset))
    print('Renewed with skip: ', max_skip)

    return construct_loader(train_dataset)


def renew_bl_loader(max_skip):
    train_dataset = VOSDataset(path.join(bl_root, 'JPEGImages'),
                               path.join(bl_root, 'Annotations'), max_skip, is_bl=True)

    print('Blender dataset size: ', len(train_dataset))
    print('Renewed with skip: ', max_skip)

    return construct_loader(train_dataset)


"""
Dataset related
"""

"""
These define the training schedule of the distance between frames
We will switch to skip_values[i] once we pass the percentage specified by increase_skip_fraction[i]
Not effective for stage 0 training
"""
skip_values = [10, 15, 20, 25, 5]

if para['stage'] == 0:
    static_root = path.expanduser(para['static_root'])
    fss_dataset = StaticTransformDataset(path.join(static_root, 'fss'), method=0)
    duts_tr_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TR'), method=1)
    duts_te_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TE'), method=1)
    ecssd_dataset = StaticTransformDataset(path.join(static_root, 'ecssd'), method=1)

    big_dataset = StaticTransformDataset(path.join(static_root, 'BIG_small'), method=1)
    hrsod_dataset = StaticTransformDataset(path.join(static_root, 'HRSOD_small'), method=1)

    # BIG and HRSOD have higher quality, use them more
    train_dataset = ConcatDataset([fss_dataset, duts_tr_dataset, duts_te_dataset, ecssd_dataset]
                                  + [big_dataset, hrsod_dataset] * 5)
    train_sampler, train_loader = construct_loader(train_dataset)

    print('Static dataset size: ', len(train_dataset))
elif para['stage'] == 1:
    increase_skip_fraction = [0.1, 0.2, 0.3, 0.4, 0.8, 1.0]
    bl_root = path.join(path.expanduser(para['bl_root']))

    train_sampler, train_loader = renew_bl_loader(5)
    renew_loader = renew_bl_loader
else:
    # stage 2 or 3
    increase_skip_fraction = [0.1, 0.2, 0.3, 0.4, 0.9, 1.0]
    # VOS dataset, 480p is used for both datasets
    # yv_root = path.join(path.expanduser(para['yv_root']), 'train_480p')
    # davis_root = path.join(path.expanduser(para['davis_root']), '2017', 'trainval')
    ASU_root = path.join(path.expanduser(para['ASU_root']))
    CVCClinic_root = path.join(path.expanduser(para['CVCClinic_root']))
    CVCColon_root = path.join(path.expanduser(para['CVCColon_root']))
    HMCQU_root = path.join(path.expanduser(para['HMCQU_root']))
    CAMUS_2CH_root = path.join(path.expanduser(para['CAMUS_2CH_root']))
    CAMUS_4CH_root = path.join(path.expanduser(para['CAMUS_4CH_root']))
    IN_HOUSE_root = path.join(path.expanduser(para['IN_HOUSE_root']))

    train_sampler, train_loader = renew_vos_loader(5)
    renew_loader = renew_vos_loader

"""
Determine current/max epoch
"""
total_epoch = math.ceil(para['iterations'] / len(train_loader))
current_epoch = total_iter // len(train_loader)
print('Number of training epochs (the last epoch might not complete): ', total_epoch)
if para['stage'] != 0:
    increase_skip_epoch = [round(total_epoch*f) for f in increase_skip_fraction]
    # Skip will only change after an epoch, not in the middle
    print('The skip value will increase approximately at the following epochs: ', increase_skip_epoch[:-1])
if not para['skip']:
    print('--------------------------Skip Deny---------------------------')

"""
Starts training
"""
# Need this to select random bases in different workers
np.random.seed(np.random.randint(2 ** 30 - 1) + local_rank * 100)
best_iou = 0
try:
    for e in range(current_epoch, total_epoch):
        print('Epoch %d/%d' % (e, total_epoch))
        if para['skip']:
            if para['stage'] != 0 and e != total_epoch and e >= increase_skip_epoch[0]:
                while e >= increase_skip_epoch[0]:
                    cur_skip = skip_values[0]
                    skip_values = skip_values[1:]
                    increase_skip_epoch = increase_skip_epoch[1:]
                print('Increasing skip to: ', cur_skip)
                train_sampler, train_loader = renew_loader(cur_skip)

        # Crucial for randomness! 
        # train_sampler.set_epoch(e)

        # Train loop
        model.train()
        list_loss = []
        list_iou = []
        list_f1_score = []
        list_dice = []
        list_precision = []
        list_recall = []
        for data in train_loader:
            total_loss,total_iou,total_f1_score,total_dice,total_precision,total_recall = model.do_pass(data, total_iter)
            total_iter += 1
            list_loss.append(total_loss)
            list_iou.append(total_iou)
            list_f1_score.append(total_f1_score)
            list_dice.append(total_dice)
            list_precision.append(total_precision)
            list_recall.append(total_recall)
            # print(f'train/loss:{"{:.5f}".format(loss)} loss_1:{"{:.5f}".format(loss_1)} loss_2:{"{:.5f}".format(loss_2)}')
            if total_iter >= para['iterations']:
                break
        mean_loss = sum(list_loss) / len(list_loss)
        mean_iou = sum(list_iou) / len(list_iou)
        mean_f1_score = sum(list_f1_score)/len(list_f1_score)
        mean_dice = sum(list_dice)/len(list_dice)
        mean_precision = sum(list_precision)/len(list_precision)
        mean_recall = sum(list_recall)/len(list_recall)
        print(
            f'mean_loss:%.5f, mean_iou:%.5f, mean_f1_score:%.5f, mean_dice:%.5f, mean_precision:%.5f, mean_recall:%.5f' % (
                mean_loss, mean_iou, mean_f1_score, mean_dice, mean_precision, mean_recall))
        if mean_iou > best_iou:
            model.save('best')
            best_iou = mean_iou
finally:
    if not para['debug'] and model.logger is not None and total_iter > 5000:
        model.save(total_iter)
    # Clean up
    # distributed.destroy_process_group()
