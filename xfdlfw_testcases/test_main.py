#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 14:46:49 2021

@author: cxue2
"""


from base.misc.metric import cross_entropy
from test_net import Linear, Inception3d
from test_data import RandomDataset

from model import Model

if __name__ == '__main__':
    
    # initialize dataset
    dst_trn = RandomDataset()
    # dst_vld = RandomDataset()
    # dst_tst = RandomDataset()
    
    # initialize model
    # net = Linear()
    # mdl = Model(net)
    
    net = Inception3d()
    mdl = Model(net)
    
    # mount model to device
    mdl.to('cuda:1')
    
    # training parameters
    kwargs_ldr_trn = {'dataset': dst_trn,
                      'batch_size': 2,
                      'shuffle': True,
                      'num_workers': 0}
    
    # kwargs_ldr_vld = {'dataset': dst_vld,
    #                   'batch_size': 10,
    #                   'shuffle': False,
    #                   'num_workers': 0}
    
    kwargs_los = {'method': 'CrossEntropyLoss'}
    
    kwargs_opt = {'method': 'Adam',
                  'lr': 1e-4}

    kwargs_msc = {'n_epochs': None,
                  'metrics_disp': ['cross_entropy',
                                   'accuracy_score'],
                  'metrics_crit': ['balanced_accuracy_score'],
                  'metrics_file': ['balanced_accuracy_score', 'cross_entropy'],
                  # 'kwargs_ldr_vld': kwargs_ldr_vld,
                  'save_mode': 0,
                  'save_dir': './test_save'}
    
    # train model
    mdl.fit(kwargs_ldr_trn, kwargs_los, kwargs_opt, **kwargs_msc)
    
    # kwargs_ldr_tst = {'dataset': dst_tst,
    #                   'batch_size': 10,
    #                   'shuffle': False,
    #                   'num_workers': 0}
    
    # test model
    # rsl = mdl.eval(kwargs_ldr_tst)
    # print(rsl.get_model_output())
    # print(rsl.get_performance_metrics(['accuracy_score']))