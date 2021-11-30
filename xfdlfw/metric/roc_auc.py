"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

from ._metric import Metric
from ._misc import _numpy
import sklearn.metrics as M


class RocAuc(Metric):

    @_numpy
    def __call__(self, output, y_true):

        return M.roc_auc_score(y_true, output[:, 1], **self.kwargs)
