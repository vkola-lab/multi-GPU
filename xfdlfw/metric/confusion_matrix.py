"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

from ._metric import Metric
from ._misc import _detach
import torch


class ConfusionMatrix(Metric):

    def __init__(self, name, n_classes, idx_output=0, idx_y_true=1):

        super().__init__(name)
        self.n_classes = n_classes
        self.idx_output = idx_output
        self.idx_y_true = idx_y_true

    @_detach
    def calc_meta(self, *operands):

        output = operands[self.idx_output]
        y_true = operands[self.idx_y_true]

        # initialize confusion matrix
        n_cls = self.n_classes
        cnf = torch.zeros(n_cls, n_cls, dtype=torch.int64, device=output.device)

        # if output and y_true are empty, return a confusion matrix of all zeros
        if output.nelement() == 0: pass

        else:
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