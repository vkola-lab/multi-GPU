#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:44:54 2021

@author: cxue2
"""

# import os, sys, inspect
# import numpy as np

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir) 

import torch
import numpy as np
from xfdlfw import Model
from xfdlfw.metric import CrossEntropy, Accuracy

class BasicConv3d(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                padding, bias=False):
        
        super(BasicConv3d, self).__init__()
        # self.module = torch.nn.Sequential(
        #     torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride,
        #             padding, bias=bias),
        #     # nn.LayerNorm(),
        #     torch.nn.BatchNorm3d(out_channels),
        #     # nn.GroupNorm(1, out_channels),
        # )

        self.module = torch.nn.Sequential(
            torch.nn.Linear(16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x):
        
        return self.module(x)

class _Dataset(torch.utils.data.Dataset):

    def __init__(self, _len):

        self._x = np.random.randn(_len, 16).astype(np.float32)
        self._y = np.random.randint(0, 2, _len).astype(np.int64)
        self._len = _len

    def __len__(self): 
        
        return self._len

    def __getitem__(self, idx):
        
        return self._x[idx], self._y[idx], idx


if __name__ == '__main__': 
    
    dst_trn = _Dataset(129)
    dst_vld = _Dataset(65)

    # dst_trn = _Dataset(128)
    # dst_vld = _Dataset(129)
    
    # initialize model
    net = BasicConv3d(1, 1, 1, 1, 0)
    mdl = Model(net)
    
    # training parameters
    kwargs_ldr_trn = {
        'dataset': dst_trn,
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 0,
        #'seed': 3227
    }
    
    kwargs_ldr_vld = {
        'dataset': dst_vld,
        'batch_size': 4,
        'shuffle': False,
        'num_workers': 0
    }

    kwargs_ldr_tst = {
        'dataset': dst_vld,
        'batch_size': 13,
        'shuffle': False,
        'num_workers': 0
    }
    
    losses = [
        torch.nn.CrossEntropyLoss()
    ]

    optimizers = [
        torch.optim.Adam(net.parameters()),
    ]

    devices = [0, 1, 2, 3]
    
    # train model
    mdl.fit(
        kwargs_ldr_trn, losses, optimizers, devices, n_epochs=8,
        kwargs_ldr_vld=kwargs_ldr_vld,
        metrics_disp=[CrossEntropy(), Accuracy()],
        metrics_crit=[Accuracy()],
        save_mode = 0,
        save_dir = './checkpoints'
    )

    # test model
    mdl.eval(
        kwargs_ldr_tst, [0],
        metrics_disp=[CrossEntropy(), Accuracy()]
    )

    #print(kwargs_ldr_trn)
    