#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:42:38 2021

@author: cxue2
"""

import sys
from tqdm import tqdm


class ProgressBar(tqdm):
    
    def __init__(self, total, desc):
        
        super().__init__(total=total, desc=desc, ascii=True, bar_format='{l_bar}{r_bar}', file=sys.stdout)
    
    def update(self, batch_size, to_disp):
        
        postfix = {}
        
        for k, v in to_disp.items():
            if k == 'cnf':           
                postfix[k] = v.__repr__().replace('\n', '')
            
            else:  
                postfix[k] = '{:.6f}'.format(v.cpu().numpy())
            
        self.set_postfix(postfix)
        super().update(batch_size)