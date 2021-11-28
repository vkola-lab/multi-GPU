#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

from abc import ABC, abstractmethod


class Metric(ABC):

    def __init__(self, name, idx_output=0, idx_y_true=1, **kwargs):
        
        self.name = name
        self.idx_y_true = idx_y_true
        self.idx_output = idx_output
        self.kwargs = kwargs

    def __call__(self, output, y_true):

        return self.from_meta(self.calc_meta(output, y_true))

    @abstractmethod
    def calc_meta(self, output, y_true): pass

    @abstractmethod
    def from_meta(self, hmp): pass

    @abstractmethod
    def join_meta(self, hmp_0, hmp_1): pass

    def compare(self, val_0, val_1):

        raise NotImplementedError

    # @property
    # def name(self):

    #     return self.name


from .confusion_matrix import ConfusionMatrix


class _Metric_00(Metric):
    ''' Dependent on confusion matrix. ''' 

    def __init__(self, name, **kwargs):

        super().__init__(name, **kwargs)
        self.cnf = ConfusionMatrix('cnf', n_classes=2)

    def calc_meta(self, output, y_true):

        return self.cnf.calc_meta(output, y_true)

    def join_meta(self, hmp_0, hmp_1):

        return self.cnf.join_meta(hmp_0, hmp_1)


class _Metric_01(Metric):
    ''' Dependent on average and # of samples. '''

    def __init__(self, name, **kwargs):

        super().__init__(name, **kwargs)

    def from_meta(self, hmp):

        return hmp['avg']

    def join_meta(self, hmp_0, hmp_1):

        return {'avg': (hmp_0['avg'] * hmp_0['n'] + hmp_1['avg'] * hmp_1['n']) / (hmp_0['n'] + hmp_1['n']),
                'n': (hmp_0['n'] + hmp_1['n'])}