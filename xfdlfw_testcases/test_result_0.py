#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

import torch
from xfdlfw import Result
from xfdlfw.metric import ConfusionMatrix, Accuracy, MeanSquaredError, MeanAbsoluteError, CrossEntropy

acc = Accuracy('acc')
ce_ = CrossEntropy('ce_')
mse = MeanSquaredError('mse')
mae = MeanAbsoluteError('mae')

# __init__
rsl = Result((acc, ce_, ce_))
print(rsl.summary())

# unregistered metric check
try:
    _ = Result((ce_, acc))
    _.summary((mse,))
except Exception as e:
    print('Exception catched:', repr(e))

# test regression
met = [mse, mae]

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
met = [ce_, acc]

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
