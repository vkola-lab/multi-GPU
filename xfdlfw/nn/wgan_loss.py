"""
Created on Thu Aug 26 15:42:38 2021

@author: cxue2
"""

import torch
import torch.nn as nn


class WGANLoss(nn.Module):

    def __init__(self, mode):

        if mode not in ('G', 'C'):
            raise RuntimeError('WGANLoss mode has to be one in (\"G\", "\C\")')

        self.mode = mode

    def forward(self, output, y_true):

        # generator loss
        if self.mode == 'C':
            l = torch.mean(output * y_true)
            
        # critic loss
        elif self.mode == 'G':
            l = -torch.mean(output[y_true == -1])

        return l

