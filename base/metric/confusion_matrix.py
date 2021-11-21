#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

from ._metric import Metric
from ._misc import _detach
import torch


class ConfusionMatrix(Metric):

    def __init__(self, n_classes):

        super().__init__(n_classes=n_classes)

    @_detach
    def calc_meta(self, output, y_true):

        # initialize confusion matrix
        n_cls = self.kwargs['n_classes']
        cnf = torch.zeros(n_cls, n_cls, dtype=torch.int64, device=output.device)

        # get predicted ys
        y_pred = torch.argmax(output, -1)

        # compute confusion matrix
        _ = torch.ones(len(output), dtype=torch.int64, device=output.device)
        cnf.index_put_((y_true, y_pred), _, accumulate=True)

        return {'cnf': cnf}

    def from_meta(self, hmp):

        return hmp['cnf']

    def join_meta(self, hmp_0, hmp_1):

        return {'cnf': hmp_0['cnf'] + hmp_1['cnf']}

    @property
    def abbr(self):

        return 'cnf'