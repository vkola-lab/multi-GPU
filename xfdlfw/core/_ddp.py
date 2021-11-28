#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:22:26 2021

@author: cxue2
"""

import os
import torch
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler


def _init_process_group(rank, world_size, master_addr='localhost', master_port='10086'):

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    init_process_group('nccl', rank=rank, world_size=world_size)


def _destroy_process_group():

    destroy_process_group()


def _create_ddp_net(net, device):

    # convert all BatchNorm*D layers in the model to torch.nn.SyncBatchNorm layers
    net_ = SyncBatchNorm.convert_sync_batchnorm(net)

    # wrap net using DistributedDataParallel
    net_ = DistributedDataParallel(net_, device_ids=[device], output_device=device)

    return net_


def _create_ddp_ldr(kwargs_ldr, rank):

    # construct kwargs for distributed sampler
    _ = DistributedSampler.__init__.__code__.co_varnames
    kwargs_smp = dict([(k, kwargs_ldr[k]) for k in _ if k in kwargs_ldr])
    kwargs_smp['rank'] = rank

    # get distributed sampler
    kwargs_ldr['sampler'] = DistributedSampler(**kwargs_smp)

    # modify kwargs for dataloader, note that sampler option is mutually exclusive with shuffle
    kwargs_ldr['shuffle'] = False
    if 'seed' in kwargs_ldr: kwargs_ldr.pop('seed')

    # create and return dataloader object
    return torch.utils.data.DataLoader(**kwargs_ldr)


class _Synchronizer:

    def __init__(self, world_size):

        self.world_size = world_size

    def all_gather(self, dat):

        rtn = []

        for r in range(self.world_size):
            rtn.append(self._retract(self._traverse(dat), rank=r))

        return rtn

    def all_gather_single_tensor(self, tensor):

        # construct placeholder for broadcast
        lst = [torch.empty_like(tensor) for _ in range(self.world_size)]

        # synchronize / broadcast
        torch.distributed.all_gather(lst, tensor)

        # return synchronized tensors
        return lst

    def _traverse(self, obj):

        if isinstance(obj, torch.Tensor):
            return self.all_gather_single_tensor(obj)

        elif isinstance(obj, list) or isinstance(obj, tuple):
            return obj.__class__([self._traverse(_) for _ in obj])

        elif isinstance(obj, dict):
            return dict(zip(obj.keys(), [self._traverse(_) for _ in obj.values()]))

        else:       
            raise RuntimeError('Unexpected type {} encountered. Please define your data structure by using only torch.Tensor, list, tuple or dict.'.format(type(obj)))

    def _retract(self, obj, rank):

        if isinstance(obj, torch.Tensor):
            return None

        elif isinstance(obj, list) or isinstance(obj, tuple):
            rtn = [self._retract(_, rank) for _ in obj]

            if None in rtn:
                return obj[rank]

            else:
                return obj.__class__(rtn)

        elif isinstance(obj, dict):
            return dict(zip(obj.keys(), [self._retract(_, rank) for _ in obj.values()]))


class _UniqueIndices:

    def __init__(self):

        self.ids = set()

    def __call__(self, ids):

        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy()

        msk = []

        for i in ids:
            if i not in self.ids:
                self.ids.add(i)
                msk.append(True)

            else:
                msk.append(False)

        return msk

    def reset(self):

        self.ids = set()

            
if __name__ == '__main__':

    to_sync = {1:2, 3:(4, 5)}
    s = _Synchronizer(2)
    print(s.all_gather(to_sync))
