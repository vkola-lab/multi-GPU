"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

import torch
from abc import ABC, abstractmethod
from ._misc import _detach


class Metric(ABC):

    def __init__(self, name):
        
        self.name = name

    def __call__(self, *operands):

        return self.from_meta(self.calc_meta(*operands))

    @abstractmethod
    def calc_meta(self, *operands): pass

    @abstractmethod
    def from_meta(self, hmp): pass

    @abstractmethod
    def join_meta(self, hmp_0, hmp_1): pass

    def compare(self, val_0, val_1):
        '''
        This method raises NotImplementedError instead of being
        decorated as abstractmethod because NotImplementedError will
        be used to indicate non-comparable metric.
        '''
        raise NotImplementedError

    # @property
    # def name(self):

    #     return self.name


from .confusion_matrix import ConfusionMatrix


class _Metric_00(Metric):
    ''' Dependent on confusion matrix. ''' 

    def __init__(self, name, idx_output=0, idx_y_true=1):

        super().__init__(name)
        self.idx_output = idx_output
        self.idx_y_true = idx_y_true
        self.cnf = ConfusionMatrix('cnf', n_classes=2)

    def calc_meta(self, *operands):

        return self.cnf.calc_meta(*operands)

    def join_meta(self, hmp_0, hmp_1):

        return self.cnf.join_meta(hmp_0, hmp_1)


class _Metric_01(Metric):
    ''' Dependent on average and number of samples. '''

    def __init__(self, name, idx_output=0, idx_y_true=1):

        super().__init__(name)
        self.idx_output = idx_output
        self.idx_y_true = idx_y_true

    @_detach
    def calc_meta(self, *operands):

        output = operands[self.idx_output]
        y_true = operands[self.idx_y_true]

        # if output and y_true are empty
        if output.nelement() == 0:
            n = torch.tensor(0, dtype=torch.int64, device=output.device)
            avg = torch.tensor(0, dtype=torch.float64, device=output.device)
            hmp = {'avg': avg, 'n': n}

        # if ouput and y_true are non-empty
        else:
            hmp = self._calc_meta(output, y_true)

        return hmp
    
    @abstractmethod
    def _calc_meta(output, y_true): pass

    def from_meta(self, hmp):

        return hmp['avg']

    def join_meta(self, hmp_0, hmp_1):

        return {'avg': (hmp_0['avg'] * hmp_0['n'] + hmp_1['avg'] * hmp_1['n']) / (hmp_0['n'] + hmp_1['n']),
                'n': (hmp_0['n'] + hmp_1['n'])}