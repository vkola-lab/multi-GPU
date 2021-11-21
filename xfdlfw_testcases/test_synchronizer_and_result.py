#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:22:26 2021

@author: cxue2
"""

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from base import Result
from base import metric
from base.core._ddp import _Synchronizer, _init_process_group, _destroy_process_group
import torch
import time

def proc(rank, world_size, devices, output, y_true, metrics):

    device = devices[rank]
    output = output[rank]
    y_true = y_true[rank]
    
    _init_process_group(rank, world_size)

    output = output.to(device)
    y_true = y_true.to(device)

    # new result
    rsl = Result(metrics)
    rsl.push(output, y_true)

    print('rank {}'.format(rank))
    print(rsl)

    # sync
    syn = _Synchronizer(world_size)
    lst_meta = syn.all_gather(rsl.summary(_key='_id', _val='meta'))

    if rank == 0:

        for meta in lst_meta[1:]:

            rsl.push_meta(meta)

        print('After sync')
        print(rsl)

    _destroy_process_group()


if __name__ == '__main__':

    devices = [0, 2, 3]
    output, y_true = [], []

    # random data
    N, C = 11, 2
    for i in range(len(devices)):
        output.append(torch.randn(N, C))
        y_true.append(torch.randint(0, C, (N,)))

    metrics = [metric.Accuracy(), metric.CrossEntropy(), metric.MatthewsCorrCoef()]

    torch.multiprocessing.spawn(proc, args=(len(devices), devices, output, y_true, metrics), nprocs=len(devices), join=True)

    output = torch.cat(output, 0)
    y_true = torch.cat(y_true, 0)

    rsl = Result.with_push(metrics, output, y_true)
    print('Non-ddp')
    print(rsl)