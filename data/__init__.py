"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import importlib
import torch.utils.data
import threading
from .vip_dataset import VipDataset
from .landscape_dataset import LandscapeDataset
from .traffic_dataset import TrafficDataset

def get_option_setter(dataset_name):
    # modify corresponding options according to different datasets.
    if dataset_name == 'vip':
        dataset_class = VipDataset
        return dataset_class.modify_commandline_options
    elif dataset_name == 'landscape':
        dataset_class = LandscapeDataset
        return dataset_class.modify_commandline_options
    elif dataset_name == 'camera_lidar_semantic':
        dataset_class = TrafficDataset
        return dataset_class.modify_commandline_options
    else:
        raise NotImplementedError


def create_dataloader(opt):
    # vip/landscape/traffic
    if opt.dataset_mode == 'vip':
        instance = VipDataset()
        instance.initialize(opt)
    elif opt.dataset_mode == 'landscape':
        instance = LandscapeDataset()
        instance.initialize(opt)
    elif opt.dataset_mode == 'traffic':
        instance = TrafficDataset()
        instance.initialize(opt)
    else:
        raise NotImplementedError
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))
    
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,  # true when train, false when test.
        num_workers=int(opt.nThreads),
        drop_last=True
    )
    return dataloader





