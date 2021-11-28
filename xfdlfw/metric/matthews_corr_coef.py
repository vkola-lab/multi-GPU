#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

from ._metric import _Metric_00
from ._misc import _fn_tpl_compare
import torch


class MatthewsCorrCoef(_Metric_00):

    def from_meta(self, hmp):

        tn, fp, fn, tp = torch.ravel(hmp['cnf'])
        n = tn + fp + fn + tp
        s = (tp + fn) / n
        p = (tp + fp) / n
        
        return (tp / n - s * p) / torch.sqrt(p * s * (1 - s) * (1 - p))

    @_fn_tpl_compare(1)
    def compare(self, val_0, val_1): pass