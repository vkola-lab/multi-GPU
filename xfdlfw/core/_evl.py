#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:42:38 2021

@author: cxue2
"""

import torch
from ._ddp import _init_process_group, _destroy_process_group
from ._ddp import _create_ddp_net, _create_ddp_ldr
from ._ddp import _Synchronizer
from ._ddp import _UniqueIndices
from ._misc import _routine_post_batch
from ..vision import ProgressBar
from ..result import Result


def _pipeline(
    rank, is_distributed, net, kwargs_ldr, devices, metrics_disp, batch_fn,
):

    # some essential local variables
    device = devices[rank]
    world_size = len(devices)

    # batch routine function
    if batch_fn is None: batch_fn = _batch_fn_default

    # initialize process group
    if is_distributed: _init_process_group(rank, world_size)
    torch.cuda.set_device(device)

    # prepare net
    net.to(device)
    if is_distributed: net = _create_ddp_net(net, device)

    # prepare dataloader
    ldr = _create_ddp_ldr(kwargs_ldr, rank) if is_distributed else torch.utils.data.DataLoader(**kwargs_ldr)

    # prepare epoch/batch result holders
    rsl_epoch = Result(metrics_disp) if rank == 0 else None
    rsl_batch = Result(metrics_disp)

    # object to check index duplications for distributed sampler
    uniq_ids = _UniqueIndices() if is_distributed else None

    # prepare result synchronizer
    syn = _Synchronizer(world_size)

    # create progress bar
    pbr = ProgressBar(len(ldr.dataset), 'Epoch --- (EVL)') if rank == 0 else None

    # set net to evaluation mode
    torch.set_grad_enabled(False)
    net.eval()

    # evaluation routine
    for batch, data in enumerate(ldr):

        batch_fn_returns = batch_fn(data[:-1], net, device)
        
        # result push, result sync, progress bar update
        _routine_post_batch(
            rank, is_distributed, device, batch_fn_returns, data[-1],
            syn, rsl_batch, rsl_epoch, uniq_ids, pbr, metrics_disp)

    # destroy process group
    if is_distributed: _destroy_process_group()


def _batch_fn_default(data, net, device):

    try:
        x, y = data

    except ValueError:
        raise RuntimeError('\"data\" cannot be unpacked into (X, Y, index).')

    # mount data to device
    x = x.to(device)
    y = y.to(device)

    # forward run
    o = net(x)

    return o, y
