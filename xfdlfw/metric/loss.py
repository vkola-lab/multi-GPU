"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

from ._metric import _Metric_01
from ._misc import _fn_tpl_compare
import torch


class Loss(_Metric_01):

    def __init__(self, name, fn, idx_output=0, idx_y_true=1):

        super().__init__(name, idx_output, idx_y_true)
        self.fn = fn

    def _calc_meta(self, output, y_true):

        n = torch.tensor(len(y_true), dtype=torch.int64, device=output.device)
        avg = self.fn(output, y_true).to(torch.float64)
        return {'avg': avg, 'n': n}

    @_fn_tpl_compare(-1)
    def compare(self, val_0, val_1): 
        
        pass