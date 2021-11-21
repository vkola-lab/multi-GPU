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

import torch
from base import Result
from base.metric import Loss, ConfusionMatrix, Accuracy, MeanSquaredError, MeanAbsoluteError, CrossEntropy

# duplication check
try:
    loss_fn_0 = torch.nn.L1Loss()
    loss_fn_1 = torch.nn.L1Loss()
    _ = Result((Loss(loss_fn_0), Loss(loss_fn_1)))
except Exception as e:
    print('Exception catched:', repr(e))

# empty result check
try:
    _ = Result((CrossEntropy(), Accuracy()))
    _.summary()
except Exception as e:
    print('Exception catched:', repr(e))

# unregistered metric check
try:
    _ = Result((CrossEntropy(), Accuracy()))
    _.summary([MeanAbsoluteError()])
except Exception as e:
    print('Exception catched:', repr(e))

# test regression
met = [MeanSquaredError(), MeanAbsoluteError()]

rsl_0 = Result(met)

o = torch.randn((7, 3))
t = torch.randn((7, 3))
rsl_0.push(o, t)

o = torch.randn((7, 3))
t = torch.randn((7, 3))
rsl_0.push(o, t)
print(rsl_0.summary(met))

rsl_1 = Result(met)

o = torch.randn((7, 3))
t = torch.randn((7, 3))
rsl_1.push(o, t)

o = torch.randn((7, 3))
t = torch.randn((7, 3))
rsl_1.push(o, t)
print(rsl_1.summary())

print('is rsl_0 better than rsl_0?', rsl_0.is_better_than(rsl_0, met))
print('is rsl_0 better than rsl_1?', rsl_0.is_better_than(rsl_1, met))

# test classification
met = [CrossEntropy(), Accuracy()]

rsl_0 = Result(met)
o = torch.randn((7, 3))
t = torch.randint(0, 3, (7,))
rsl_0.push(o, t)
o = torch.randn((7, 3))
t = torch.randint(0, 3, (7,))
rsl_0.push(o, t)
print(rsl_0.summary())

rsl_1 = Result(met)
o = torch.randn((7, 3))
t = torch.randint(0, 3, (7,))
rsl_1.push(o, t)
o = torch.randn((7, 3))
t = torch.randint(0, 3, (7,))
rsl_1.push(o, t)
print(rsl_1.summary())

print('is rsl_0 better than rsl_1?', rsl_0.is_better_than(rsl_1, met))
