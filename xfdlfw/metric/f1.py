#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

from ._metric import _Metric_00
from ._misc import _fn_tpl_compare
import torch


class F1(_Metric_00):

    def __init__(self):

        super().__init__()

    def from_meta(self, hmp):

        _, fp, fn, tp = torch.ravel(hmp['cnf'])
        prc = tp / (tp + fp)
        sen = tp / (tp + fn)

        return 2 * (prc * sen) / (prc + sen)

    @_fn_tpl_compare(1)
    def compare(self, val_0, val_1): pass

    @property
    def abbr(self):

        return 'f1s'