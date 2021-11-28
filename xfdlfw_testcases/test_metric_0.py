#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

import numpy as np
import torch
from xfdlfw.metric import ConfusionMatrix, Accuracy, Precision, Recall, F1, CrossEntropy, MatthewsCorrCoef, MeanAbsoluteError, MeanSquaredError

# test classification
a = np.random.randn(128, 2).astype(np.float32)
b = np.random.randint(0, 2, 128).astype(np.int64)

a_ = torch.tensor(a).to(0)
b_ = torch.tensor(b).to(0)

# cnf = ConfusionMatrix(3)
# print(cnf(a_, b_))
# print(sk.metrics.confusion_matrix(np.argmax(a, -1), b))

hmp_cls = {
    'cnf': ConfusionMatrix,
    'acc': Accuracy,
    'prc': Precision,
    'rcl': Recall,
    'f1s': F1,
    'mcc': MatthewsCorrCoef,
    'ce_': CrossEntropy
}

for k, cls in hmp_cls.items():

    obj = cls(k) if cls is not ConfusionMatrix else cls(k, n_classes=2)
    print(obj.name, obj(a_, b_))

    hmp = obj.calc_meta(a_, b_)
    print('{}_calc_meta'.format(obj.name), hmp)
    print('{}_from_meta'.format(obj.name), obj.from_meta(hmp))
    print('{}_join_meta'.format(obj.name), obj.join_meta(hmp, hmp))
    print()

# test regression
a = np.random.randn(2, 182, 218, 182).astype(np.float32)
b = np.random.randn(2, 182, 218, 182).astype(np.float32)

a_ = torch.tensor(a).to(0)
b_ = torch.tensor(b).to(0)

hmp_cls = {
    'mse': MeanSquaredError,
    'mae': MeanAbsoluteError,
}

for k, cls in hmp_cls.items():

    obj = cls(k)
    print(obj.name, obj(a_, b_))

    hmp = obj.calc_meta(a_, b_)
    print('{}_calc_meta'.format(obj.name), hmp)
    print('{}_from_meta'.format(obj.name), obj.from_meta(hmp))
    print('{}_join_meta'.format(obj.name), obj.join_meta(hmp, hmp))
    print()