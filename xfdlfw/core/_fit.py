#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:42:38 2021

@author: cxue2
"""

import torch
from . import _evl
from ._ddp import _init_process_group
from ._ddp import _destroy_process_group
from ._ddp import _create_ddp_net
from ._ddp import _create_ddp_ldr
from ._ddp import _Synchronizer
from ._ddp import _UniqueIndices
from ._misc import _routine_post_batch
from ._misc import _routine_post_epoch
from ..result import Result
from ..vision import ProgressBar
from itertools import count
from torch.utils.tensorboard import SummaryWriter


def _pipeline(
    rank, is_distributed, net, queue, event,
    kwargs_ldr_trn, losses, optimizers, devices,
    n_epochs, kwargs_ldr_vld, save_mode, save_dir,
    metrics_disp, metrics_crit,
    batch_fn_trn, batch_fn_vld,
    tensorboard_fn
):

    _input_check(**locals())

    # some essential local variables
    device = devices[rank]
    world_size = len(devices)
    metrics = metrics_disp + metrics_crit

    # initialize process group
    if is_distributed: _init_process_group(rank, world_size)
    torch.cuda.set_device(device)

    # prepare batch routine function
    if batch_fn_trn is None: batch_fn_trn =      _batch_fn_default
    if batch_fn_vld is None: batch_fn_vld = _evl._batch_fn_default

    # tensorboard writer for visualization
    tb_writer = None if (tensorboard_fn is None) or (rank != 0) else SummaryWriter()
    
    # prepare net
    net.to(device)
    if is_distributed: net = _create_ddp_net(net, device)
    
    # prepare dataloaders
    ldr_trn = None if kwargs_ldr_trn is None else (
        _create_ddp_ldr(kwargs_ldr_trn, rank) if is_distributed else torch.utils.data.DataLoader(**kwargs_ldr_trn)
    )
    ldr_vld = None if kwargs_ldr_vld is None else (
        _create_ddp_ldr(kwargs_ldr_vld, rank) if is_distributed else torch.utils.data.DataLoader(**kwargs_ldr_vld)
    )

    # prepare epoch/batch result holders
    rsl_epoch_best = Result(metrics) if rank == 0 else None
    rsl_epoch = Result(metrics) if rank == 0 else None
    rsl_batch = Result(metrics)

    # object to check index duplications for distributed sampler
    uniq_ids = _UniqueIndices() if is_distributed else None

    # prepare result synchronizer
    syn = _Synchronizer(world_size)

    # epoch loop
    for epoch in range(n_epochs) if n_epochs is not None else count():
        # set net to training mode
        torch.set_grad_enabled(True)
        net.train()

        # distributed sampler must be reset for randomness at each epoch
        if is_distributed: ldr_trn.sampler.set_epoch(epoch)

        # create progress bar
        pbr = ProgressBar(len(ldr_trn.dataset), 'Epoch {:03d} (TRN)'.format(epoch)) if rank == 0 else None

        # mini-batch training loop
        for batch, data in enumerate(ldr_trn):
            batch_fn_returns = batch_fn_trn(data[:-1], net, device, losses, optimizers, epoch, batch)

            # result push, result sync, index duplication removal, progress bar update
            _routine_post_batch(
                rank, is_distributed, device, batch_fn_returns, data[-1],
                syn, rsl_batch, rsl_epoch, uniq_ids, pbr, metrics_disp)

        # set net to evaluation mode for tensorboard visualization and validation
        torch.set_grad_enabled(False)
        net.eval()

        # tensorboard
        if tb_writer is not None:
            tensorboard_fn(
                tb_writer, 'trn', device, 
                net.module if is_distributed else net, 
                ldr_trn, rsl_epoch, epoch
            )

        # reset epoch result holder and uniq_ids
        if rank == 0: rsl_epoch.reset()
        if is_distributed: uniq_ids.reset()

        # validate or not?
        if ldr_vld is None: continue

        # create progress bar
        pbr = ProgressBar(len(ldr_vld.dataset), 'Epoch --- (EVL)') if rank == 0 else None

        # validation
        for batch, data in enumerate(ldr_vld):
            batch_fn_returns = batch_fn_vld(data[:-1], net, device)

            # result push, result sync, progress bar update
            _routine_post_batch(
                rank, is_distributed, device, batch_fn_returns, data[-1],
                syn, rsl_batch, rsl_epoch, uniq_ids, pbr, metrics_disp
            )

        # validation performance compare, model save
        _routine_post_epoch(rank, is_distributed, net, rsl_epoch, rsl_epoch_best, epoch, metrics_crit, save_mode, save_dir)

        # tensorboard
        if tb_writer is not None:
            tensorboard_fn(
                tb_writer, 'vld', device, 
                net.module if is_distributed else net, 
                ldr_vld, rsl_epoch, epoch
            )

        # reset epoch result holder
        if rank == 0: rsl_epoch.reset()
        if is_distributed: uniq_ids.reset()

    # send model back to the main process
    if is_distributed and rank == 0: queue.put(net.module.cpu().state_dict())

    # block until model transfer is done
    if is_distributed: event.wait()

    # destroy process group
    if is_distributed: _destroy_process_group()


def _batch_fn_default(data, net, device, losses, optimizers, epoch, batch):

    try:
        x, y = data

    except ValueError:
        raise RuntimeError('\"data\" cannot be unpacked into (X, Y, index).')
 
    # clean gradients
    optimizers[0].zero_grad()

    # mount data to device
    x = x.to(device)
    y = y.to(device)

    # forward run
    o = net(x)

    # calculate loss
    l = losses[0](o, y)

    # backward run and update parameters
    l.backward()
    optimizers[0].step()

    return o, y
    
    
def _input_check(**kwargs):

    if len(kwargs['losses']) > 1 and (kwargs['batch_fn_trn'] is None or kwargs['batch_fn_vld'] is None):

        raise RuntimeError('Default batch training routine cannot handle multiple losses.')

    if len(kwargs['optimizers']) > 1 and (kwargs['batch_fn_trn'] is None or kwargs['batch_fn_vld'] is None):

        raise RuntimeError('Default batch training routine cannot handle multiple optimizers.')

    if kwargs['save_mode'] not in [0, 1, 2]:
        
        raise RuntimeError('\"{}\" is not a valid <save_mode>.'.format(kwargs['save_mode']))
        
    if kwargs['save_mode'] == 1 and kwargs['save_dir'] is None:
        
        raise RuntimeError('If <save_mode> == 1, <save_dir> must be specified.')
        
    if kwargs['save_mode'] == 2 and (kwargs['kwargs_ldr_vld'] is None or 
                                     kwargs['save_dir'] is None or 
                                     kwargs['metrics_crit'] is None):
        
        raise RuntimeError('If <save_mode> is 2, <save_dir>, <kwargs_ldr_vld> and <metrics_crit> must be specified.')
