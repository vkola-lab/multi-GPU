#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

from ._metric import _Metric_01
from ._misc import _detach, _fn_tpl_compare
import torch


class MeanAbsoluteError(_Metric_01):

    @_detach
    def calc_meta(self, output, y_true):

        # if output and y_true are empty
        if output.nelement() == 0:
            n = torch.tensor(0, dtype=torch.int64, device=output.device)
            avg = torch.tensor(0, dtype=torch.float64, device=output.device)

        # if ouput and y_true are non-empty
        else:
            n = torch.tensor(len(y_true), dtype=torch.int64, device=output.device)
            avg = torch.nn.functional.l1_loss(output, y_true).to(torch.float64)

        return {'avg': avg, 'n': n}

    @_fn_tpl_compare(-1)
    def compare(self, val_0, val_1): pass