#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

import torch
import torch.optim as optim

model = torch.nn.Linear(2, 2) 

# Initialize optimizer
optimizer = optim.Adam((), lr=0.001)

extra_params = torch.randn(2, 2)
optimizer.param_groups.append({'params': extra_params })

#then you can print your `extra_params`
print("extra params", extra_params)
print("optimizer params", optimizer.param_groups)