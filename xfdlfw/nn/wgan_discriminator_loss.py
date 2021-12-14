"""
Created on Thu Aug 26 15:42:38 2021

@author: cxue2
"""

import torch


class WGANDiscriminatorLoss(torch.nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, output, y_true):

        loss_real = -torch.mean(output[y_true == 1])
        loss_fake =  torch.mean(output[y_true == 0])
        return loss_real + loss_fake


if __name__ == '__main__':

    loss = WGANDiscriminatorLoss()

    output = torch.randn(10)
    y_true = torch.ones(10)
    y_true[:5] = 0

    print(loss(output, y_true))

