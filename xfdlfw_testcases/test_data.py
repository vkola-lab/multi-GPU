# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 11:25:20 2021

@author: Iluva
"""


import torch
import numpy as np


class RandomDataset(torch.utils.data.Dataset):
    
    
    def __init__(self):
        
        self.x = np.random.randn(100, 1, 182, 218, 182).astype(np.float32)
        self.y = np.random.randint(0, 2, 100).astype(np.int64)
    
    
    def __len__(self):
        
        return len(self.x)
    
    
    def __getitem__(self, idx):
        
        return self.x[idx, :, :, :, :], self.y[idx]