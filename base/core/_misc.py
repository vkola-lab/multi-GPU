#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:42:38 2021

@author: cxue2
"""

import torch
from datetime import datetime


def _routine_post_batch(
    rank, is_distributed, device, output, y_true, ids,
    syn, rsl_batch, rsl_epoch, uniq_ids, pbr, metrics_disp
):

    # remove index duplication which is caused by distributed sampler
    if is_distributed:
        # check sample index duplications
        msk = torch.tensor(uniq_ids(ids), dtype=torch.bool, device=device)

        # sync indices and update unique_ids object
        lst_ids = syn.all_gather_single_tensor(ids.to(device))
        for _ids in lst_ids:
            # push to uniq_ids obj
            uniq_ids(_ids)

        # sync the masks of index uniqueness
        lst_msk = syn.all_gather_single_tensor(msk)
        n_uniq_ids = sum([torch.sum(_msk) for _msk in lst_msk])

    if is_distributed:
        # push local outputs (duplication removed) to batch result holder
        rsl_batch.push(output[msk], y_true[msk])

        # sync metric meta
        lst_meta = syn.all_gather(rsl_batch.summary(_key='_id', _val='meta'))

    else:
        # push local outputs to batch result holder
        rsl_batch.push(output, y_true)

        # no need to sync
        lst_meta = [rsl_batch.summary(_key='_id', _val='meta')]

    rsl_batch.reset()

    # merge batch result to epoch result, then read metrics to show on progress bar
    if rank == 0:
        for meta in lst_meta: rsl_epoch.push_meta(meta)
        to_disp = rsl_epoch.summary(metrics_disp, _tuple=True)

        if is_distributed:
            pbr.update(n_uniq_ids.item(), to_disp)

        else:
            pbr.update(len(output), to_disp)


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
            rsl_epoch_best.push_meta(rsl_epoch.summary(_key='_id', _val='meta'))

            # save model
            torch.save(net_.state_dict(), fname)
