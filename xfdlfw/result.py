"""
Created on Fri Aug 27 14:47:38 2021

@author: cxue2
"""

import torch
import itertools


class Result:
    
    def __init__(self, metrics):

        # register metrics
        self._hmp = {}
        for met_obj in metrics:
            if met_obj.name not in self._hmp:
                self._hmp[met_obj.name] = [met_obj, None]


    @classmethod
    def with_push(cls, metrics, *operands):

        rsl_obj = cls(metrics)
        rsl_obj.push(*operands)
        return rsl_obj
    
    
    def push(self, *operands):
        
        for p in self._hmp.values():

            # calcualte meta
            meta = p[0].calc_meta(*operands)

            # join meta
            p[1] = meta if p[1] is None else p[0].join_meta(p[1], meta)


    def push_meta(self, hmp):

        for name in hmp:
            p, meta = self._hmp[name], hmp[name]
            p[1] = meta if p[1] is None else p[0].join_meta(p[1], meta)


    def is_empty(self):

        # return true (i.e. is empty) if there is any meta being None
        for p in self._hmp.values():
            if p[1] is None:
                return True
        return False


    def reset(self):

        for p in self._hmp.values():
            p[1] = None
  
    
    def summary(self, metrics=None, _val='value'):

        if metrics is None:
            metrics = [p[0] for p in self._hmp.values()]

        # metric name list
        lst_k = [met_obj.name for met_obj in metrics]

        # val list
        rtn = dict()
        for name in lst_k:
            if self._hmp[name][1] is None:
                rtn[name] = None

            elif _val == 'value':
                rtn[name] = self._hmp[name][0].from_meta(self._hmp[name][1])

            elif _val == 'meta':
                rtn[name] = self._hmp[name][1]

        return rtn

    
    def is_better_than(self, tar, metrics):

        # calculate metrics
        hmp_0 = self.summary(metrics)
        hmp_1 =  tar.summary(metrics)

        for name in hmp_0:
            # compare
            cmp = self._hmp[name][0].compare(hmp_0[name], hmp_1[name])
            
            if cmp == 1: 
                return True

            elif cmp == -1:
                return False
        
        return False


    def __repr__(self):

        str_ = ''

        for k, v in self.summary():
            str_ += '{}: {}\n'.format(k, v.cpu().numpy())

        return str_