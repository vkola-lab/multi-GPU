#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 14:47:38 2021

@author: cxue2
"""

import torch
import torch.nn as nn



class Linear(nn.Module):
    
    
    def __init__(self):
        
        super(Linear, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 2))
        
        
    def forward(self, x):
        
        return self.module(x)



class BasicConv3d(nn.Module):
    
    
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, bias=False):
        
        super(BasicConv3d, self).__init__()
        self.module = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride,
                          padding, bias=bias),
                # nn.LayerNorm(),
                nn.BatchNorm3d(out_channels),
                # nn.GroupNorm(1, out_channels),
                )
        
        
    def forward(self, x):
        
        return self.module(x)
    
    
    
class InceptionA(nn.Module):
    
    
    def __init__(self, in_channels, out_channels_branch_pool):
        
        super(InceptionA, self).__init__()
        
        self.branch_1x1x1 = BasicConv3d(in_channels, 64, 1, 1, 0)
        
        self.branch_5x5x5_1 = BasicConv3d(in_channels, 48, 1, 1, 0)
        self.branch_5x5x5_2 = BasicConv3d(48, 64, 5, 1, 2)
        
        self.branch_3x3x3dbl_1 = BasicConv3d(in_channels, 64, 1, 1, 0)
        self.branch_3x3x3dbl_2 = BasicConv3d(64, 96, 3, 1, 1)
        self.branch_3x3x3dbl_3 = BasicConv3d(96, 96, 3, 1, 1)
        
        self.branch_pool_1 = nn.AvgPool3d(3, 1, 1)
        self.branch_pool_2 = BasicConv3d(in_channels, out_channels_branch_pool, 1, 1, 0)
        
        
    def forward(self, x):
        
        out_1 = self.branch_1x1x1(x)
        
        out_2 = self.branch_5x5x5_1(x)
        out_2 = self.branch_5x5x5_2(out_2)
        
        out_3 = self.branch_3x3x3dbl_1(x)
        out_3 = self.branch_3x3x3dbl_2(out_3)
        out_3 = self.branch_3x3x3dbl_3(out_3)
        
        out_p = self.branch_pool_1(x)
        out_p = self.branch_pool_2(out_p)
        
        return torch.cat([out_1, out_2, out_3, out_p], 1)
    


class InceptionB(nn.Module):
    
    
    def __init__(self, in_channels):
        
        super(InceptionB, self).__init__()
        
        self.branch_3x3x3 = BasicConv3d(in_channels, 384, 3, 2, 0)
        
        self.branch_3x3x3dbl_1 = BasicConv3d(in_channels, 64, 1, 1, 0)
        self.branch_3x3x3dbl_2 = BasicConv3d(64, 96, 3, 1, 1)
        self.branch_3x3x3dbl_3 = BasicConv3d(96, 96, 3, 2, 0)
        
        self.maxpool = nn.MaxPool3d(3, 2, 0)
        
        
    def forward(self, x):
        
        out_1 = self.branch_3x3x3(x)
        
        out_2 = self.branch_3x3x3dbl_1(x)
        out_2 = self.branch_3x3x3dbl_2(out_2)
        out_2 = self.branch_3x3x3dbl_3(out_2)
        
        out_3 = self.maxpool(x)
        
        return torch.cat([out_1, out_2, out_3], 1)



class InceptionC(nn.Module):
    
    
    def __init__(self, in_channels, out_channels_branch_pool, conv_channels):
        
        super(InceptionC, self).__init__()
        
        self.branch_1x1x1 = BasicConv3d(in_channels, 192, 1, 1, 0)
        
        self.branch_7x7x7_1 = BasicConv3d(in_channels, conv_channels, 1, 1, 0)
        self.branch_7x7x7_2 = BasicConv3d(conv_channels, conv_channels, (7, 1, 1), 1, (3, 0, 0))
        self.branch_7x7x7_3 = BasicConv3d(conv_channels, conv_channels, (1, 7, 1), 1, (0, 3, 0))
        self.branch_7x7x7_4 = BasicConv3d(conv_channels, 192, (1, 1, 7), 1, (0, 0, 3))
        
        self.branch_7x7x7dbl_1 = BasicConv3d(in_channels, conv_channels, 1, 1, 0)
        self.branch_7x7x7dbl_2 = BasicConv3d(conv_channels, conv_channels, (7, 1, 1), 1, (3, 0, 0))
        self.branch_7x7x7dbl_3 = BasicConv3d(conv_channels, conv_channels, (1, 7, 1), 1, (0, 3, 0))
        self.branch_7x7x7dbl_4 = BasicConv3d(conv_channels, conv_channels, (1, 1, 7), 1, (0, 0, 3))
        self.branch_7x7x7dbl_5 = BasicConv3d(conv_channels, conv_channels, (7, 1, 1), 1, (3, 0, 0))
        self.branch_7x7x7dbl_6 = BasicConv3d(conv_channels, conv_channels, (1, 7, 1), 1, (0, 3, 0))
        self.branch_7x7x7dbl_7 = BasicConv3d(conv_channels, 192, (1, 1, 7), 1, (0, 0, 3))
        
        self.branch_pool_1 = nn.AvgPool3d(3, 1, 1)
        self.branch_pool_2 = BasicConv3d(in_channels, out_channels_branch_pool, 1, 1, 0)
        
        
    def forward(self, x):
        
        out_1 = self.branch_1x1x1(x)
        
        out_2 = self.branch_7x7x7_1(x)
        out_2 = self.branch_7x7x7_2(out_2)
        out_2 = self.branch_7x7x7_3(out_2)
        out_2 = self.branch_7x7x7_4(out_2)
        
        out_3 = self.branch_7x7x7dbl_1(x)
        out_3 = self.branch_7x7x7dbl_2(out_3)
        out_3 = self.branch_7x7x7dbl_3(out_3)
        out_3 = self.branch_7x7x7dbl_4(out_3)
        out_3 = self.branch_7x7x7dbl_5(out_3)
        out_3 = self.branch_7x7x7dbl_6(out_3)
        out_3 = self.branch_7x7x7dbl_7(out_3)
        
        out_p = self.branch_pool_1(x)
        out_p = self.branch_pool_2(out_p)
        
        return torch.cat([out_1, out_2, out_3, out_p], 1)
    
    
    
class InceptionAux(nn.Module):
    
    
    def __init__(self):
        
        super(InceptionAux, self).__init__()
        
        
    def forward(self, x):
        
        pass
    
    

class InceptionD(nn.Module):
    
    
    def __init__(self, in_channels):
        
        super(InceptionD, self).__init__()
        
        self.branch_3x3x3_1 = BasicConv3d(in_channels, 192, 1, 1, 0)
        self.branch_3x3x3_2 = BasicConv3d(192, 320, 3, 2, 0)
        
        self.branch_7x7x7_1 = BasicConv3d(in_channels, 192, 1, 1, 0)
        self.branch_7x7x7_2 = BasicConv3d(192, 192, (7, 1, 1), 1, (3, 0, 0))
        self.branch_7x7x7_3 = BasicConv3d(192, 192, (1, 7, 1), 1, (0, 3, 0))
        self.branch_7x7x7_4 = BasicConv3d(192, 192, (1, 1, 7), 1, (0, 0, 3))
        self.branch_7x7x7_5 = BasicConv3d(192, 192, 3, 2, 0)
        
        self.maxpool = nn.MaxPool3d(3, 2, 0)
        
        
    def forward(self, x):
        
        out_1 = self.branch_3x3x3_1(x)
        out_1 = self.branch_3x3x3_2(out_1)
        
        out_2 = self.branch_7x7x7_1(x)
        out_2 = self.branch_7x7x7_2(out_2)
        out_2 = self.branch_7x7x7_3(out_2)
        out_2 = self.branch_7x7x7_4(out_2)
        out_2 = self.branch_7x7x7_5(out_2)
    
        out_p = self.maxpool(x)
      
        return torch.cat([out_1, out_2, out_p], 1)
    


class InceptionE(nn.Module):
    
    
    def __init__(self, in_channels):
        
        super(InceptionE, self).__init__()
        
        self.branch_1x1x1 = BasicConv3d(in_channels, 320, 1, 1, 0)
        
        self.branch_3x3x3_1 = BasicConv3d(in_channels, 256, 1, 1, 0)
        self.branch_3x3x3_2a = BasicConv3d(256, 256, (3, 1, 1), 1, (1, 0, 0))
        self.branch_3x3x3_2b = BasicConv3d(256, 256, (1, 3, 1), 1, (0, 1, 0))
        self.branch_3x3x3_2c = BasicConv3d(256, 256, (1, 1, 3), 1, (0, 0, 1))
        
        self.branch_3x3x3dbl_1 = BasicConv3d(in_channels, 448, 1, 1, 0)
        self.branch_3x3x3dbl_2 = BasicConv3d(448, 256, 3, 1, 1)
        self.branch_3x3x3dbl_3a = BasicConv3d(256, 256, (3, 1, 1), 1, (1, 0, 0))
        self.branch_3x3x3dbl_3b = BasicConv3d(256, 256, (1, 3, 1), 1, (0, 1, 0))
        self.branch_3x3x3dbl_3c = BasicConv3d(256, 256, (1, 1, 3), 1, (0, 0, 1))
        
        self.branch_pool_1 = nn.AvgPool3d(3, 1, 1)
        self.branch_pool_2 = BasicConv3d(in_channels, 192, 1, 1, 0)
        
        
    def forward(self, x):
        
        out_1 = self.branch_1x1x1(x)
        
        out_2 = self.branch_3x3x3_1(x)
        out_2a = self.branch_3x3x3_2a(out_2)
        out_2b = self.branch_3x3x3_2b(out_2)
        out_2c = self.branch_3x3x3_2c(out_2)
        out_2 = torch.cat([out_2a, out_2b, out_2c], 1)
        
        out_3 = self.branch_3x3x3dbl_1(x)
        out_3 = self.branch_3x3x3dbl_2(out_3)
        out_3a = self.branch_3x3x3dbl_3a(out_3)
        out_3b = self.branch_3x3x3dbl_3b(out_3)
        out_3c = self.branch_3x3x3dbl_3c(out_3)
        out_3 = torch.cat([out_3a, out_3b, out_3c], 1)
        
        out_p = self.branch_pool_1(x)
        out_p = self.branch_pool_2(out_p)
    
        return torch.cat([out_1, out_2, out_3, out_p], 1)
        
          

class Inception3d(nn.Module):
    
    
    def __init__(self):
        
        super(Inception3d, self).__init__()
        
        self.conv3d_1a = BasicConv3d(1, 32, 3, 2, 0)        
        self.conv3d_2a = BasicConv3d(32, 32, 3, 1, 0)
        self.conv3d_2b = BasicConv3d(32, 64, 3, 1, 1)
        self.maxpool_1 = nn.MaxPool3d(3, 2, 0)
        self.conv3d_3b = BasicConv3d(64, 80, 1, 1, 0)
        self.conv3d_4a = BasicConv3d(80, 192, 3, 1, 0)
        self.maxpool_2 = nn.MaxPool3d(3, 2, 0)
        
        self.mixed_5b = InceptionA(192, 32)
        self.mixed_5c = InceptionA(256, 64)
        self.mixed_5d = InceptionA(288, 64)
        
        self.mixed_6a = InceptionB(288)
        self.mixed_6b = InceptionC(768, 192, 128)
        self.mixed_6c = InceptionC(768, 192, 160)
        self.mixed_6d = InceptionC(768, 192, 160)
        self.mixed_6e = InceptionC(768, 192, 192)
        
        self.mixed_7a = InceptionD(768)
        self.mixed_7b = InceptionE(1280)
        self.mixed_7c = InceptionE(2048)
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten(1, -1)
        self.dropout = nn.Dropout(.5)
        self.fc = nn.Linear(2048, 2)
        
        self.module = nn.Sequential(
                self.conv3d_1a,
                self.conv3d_2a,
                self.conv3d_2b,
                self.maxpool_1,
                self.conv3d_3b,
                self.conv3d_4a,
                self.maxpool_2,
                self.mixed_5b,
                self.mixed_5c,
                self.mixed_5d,
                self.mixed_6a,
                self.mixed_6b,
                self.mixed_6c,
                self.mixed_6d,
                self.mixed_6e,
                self.mixed_7a,
                self.mixed_7b,
                self.mixed_7c,
                self.avgpool,
                self.flatten,
                self.dropout,
                self.fc)
        
        
    
    def forward(self, x):
        
        return self.module(x)



if __name__ == '__main__':
    
    import tqdm
    
    x = torch.rand(2, 1, 182, 218, 182).to('cuda:0')
    mdl = Inception3d().to('cuda:0')
    
    mdl.train()
    for i in tqdm.tqdm(range(1000)):
        y = mdl(x)
    
    print(mdl(x).shape)
    
#    from torchsummary import summary
    # summary(mdl, (1, 182, 218, 182))