#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

import os
import sys
import inspect
import sklearn as sk

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
import torch
from base.metric import ConfusionMatrix, Accuracy, Precision, Recall, F1, CrossEntropy, MatthewsCorrCoef, MeanAbsoluteError, MeanSquaredError

# test classification
a = np.random.randn(128, 2).astype(np.float32)
b = np.random.randint(0, 2, 128).astype(np.int64)

a_ = torch.tensor(a).to(0)
b_ = torch.tensor(b).to(0)

# cnf = ConfusionMatrix(3)
# print(cnf(a_, b_))
# print(sk.metrics.confusion_matrix(np.argmax(a, -1), b))

for cls in (ConfusionMatrix, Accuracy, Precision, Recall, F1, MatthewsCorrCoef, CrossEntropy):

    obj = cls() if cls is not ConfusionMatrix else cls(n_classes=2)
    print(type(obj.compare))
    print(obj.abbr, obj(a_, b_))

    hmp = obj.calc_meta(a_, b_)
    print('{}_calc_meta'.format(obj.abbr), hmp)
    print('{}_from_meta'.format(obj.abbr), obj.from_meta(hmp))
    print('{}_join_meta'.format(obj.abbr), obj.join_meta(hmp, hmp))
    print()

# test regression
a = np.random.randn(2, 182, 218, 182).astype(np.float32)
b = np.random.randn(2, 182, 218, 182).astype(np.float32)

a_ = torch.tensor(a).to(0)
b_ = torch.tensor(b).to(0)

for cls in (MeanSquaredError, MeanAbsoluteError):

    obj = cls()
    print(obj.abbr, obj(a_, b_))

    hmp = obj.calc_meta(a_, b_)
    print('{}_calc_meta'.format(obj.abbr), hmp)
    print('{}_from_meta'.format(obj.abbr), obj.from_meta(hmp))
    print('{}_join_meta'.format(obj.abbr), obj.join_meta(hmp, hmp))
    print()

# test _id
cnf_0 = ConfusionMatrix(2)
cnf_1 = ConfusionMatrix(2)
cnf_2 = ConfusionMatrix(3)

print('cnf_0._id == cnf_1._id?', cnf_0._id == cnf_1._id)
print('cnf_1._id == cnf_2._id?', cnf_1._id == cnf_2._id)
