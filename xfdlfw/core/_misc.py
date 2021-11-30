#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:42:38 2021

@author: cxue2
"""

import torch
from datetime import datetime


def _routine_post_batch(
    rank, is_distributed, device, batch_fn_returns, ids,
    syn, rsl_batch, rsl_epoch, uniq_ids, pbr, metrics_disp
):

    # manage indices
    if is_distributed:
        '''
        Each process maintains a set of unique sample indices independently, but all
        sets are supposed to be identical using broadcasting after each batch. 
        '''
        lst_ids = syn.all_gather_single_tensor(ids.to(device))
        lst_msk = []
        
        # record index uniqueness mask for all ranks
        for rank_, ids_ in enumerate(lst_ids):
            if rank == rank_:
                # index uniqueness mask for local rank
                msk = uniq_ids(ids_)
                lst_msk.append(msk)

            else:
                lst_msk.append(uniq_ids(ids_))

        # the num. of unique indice equals the num. of Trues in lst_msk
        n_uniq_ids = sum([sum(msk_) for msk_ in lst_msk])

    else:
        '''
        The default sampler for non-distributed run is assumed to yield unique indice.
        '''
        n_uniq_ids = len(ids)

    # manage results
    if is_distributed:

        # apply mask to remove duplications
        batch_fn_returns = list(batch_fn_returns)
        for i, obj in enumerate(batch_fn_returns):
            if isinstance(obj, torch.Tensor) and len(obj) == len(msk):
                batch_fn_returns[i] = obj[msk]

        # push local outputs (duplications removed) to batch result holder
        rsl_batch.push(*batch_fn_returns)

        # sync metric meta
        lst_meta = syn.all_gather(rsl_batch.summary(_val='meta'))

    else:
        # push local outputs to batch result holder
        rsl_batch.push(*batch_fn_returns)

        # no need to sync
        lst_meta = [rsl_batch.summary(_val='meta')]

    rsl_batch.reset()

    # merge batch result to epoch result, then read metrics to show on progress bar
    if rank == 0:
        for meta in lst_meta:
            rsl_epoch.push_meta(meta)
            
        to_disp = rsl_epoch.summary(metrics_disp)
        pbr.update(n_uniq_ids, to_disp)


def _routine_post_epoch(
    rank, is_distributed, net, rsl_epoch, rsl_epoch_best, epoch,
    metrics_crit, save_mode, save_dir
):

    if rank != 0 or save_mode == 0: return

    # filename
    str_datetime = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    fname = '{}/{}_{}.pt'.format(save_dir, epoch, str_datetime)

    # the actual net is pointed by ddp_net.module
    net_ = net.module if is_distributed else net

    # save regardless of any condition
    if save_mode == 1:
        torch.save(net_.state_dict(), fname)

    # save iff the current epoch is the best in terms of metrics_crit
    elif save_mode == 2:
        # either empty, or no longer the best
        if rsl_epoch_best.is_empty() or (not rsl_epoch_best.is_better_than(rsl_epoch, metrics_crit)):
            # replace rsl_epoch_best for the current
            rsl_epoch_best.reset()
            rsl_epoch_best.push_meta(rsl_epoch.summary(_val='meta'))

            # save model
            torch.save(net_.state_dict(), fname)
