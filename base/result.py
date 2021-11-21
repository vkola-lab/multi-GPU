"""
Created on Fri Aug 27 14:47:38 2021

@author: cxue2
"""

import torch
import itertools


class Result:
    
    def __init__(self, metrics):

        # get metric ids
        ids = [_._id for _ in metrics]

        # register metrics
        self._hmp = {}

        for i, id_ in enumerate(ids):
            self._hmp[id_] = [metrics[i], None]


    @classmethod
    def with_push(cls, metrics, output, y_true):

        obj = cls(metrics)
        obj.push(output, y_true)
        return obj
    
    
    def push(self, output, y_true):
        
        for p in self._hmp.values():
            # calcualte meta
            meta = p[0].calc_meta(output, y_true)

            # join meta
            p[1] = meta if p[1] is None else p[0].join_meta(p[1], meta)


    def push_meta(self, hmp):

        for id_ in hmp:
            p, meta = self._hmp[id_], hmp[id_]
            p[1] = meta if p[1] is None else p[0].join_meta(p[1], meta)


    def is_empty(self):

        # return true (i.e. is empty) if there is any meta being None
        for p in self._hmp.values():
            if p[1] is None:
                return True
        
        return False


    def reset(self):

        for id_ in self._hmp:
            self._hmp[id_][1] = None
  
    
    def summary(self, metrics=None, _key='abbr', _val='value', _tuple=False):

        # assert _key in ('_id', 'abbr')
        # assert _val in ('meta', 'value')

        if metrics is None:
            metrics = [p[0] for p in self._hmp.values()]

        # key list
        lst_k = [getattr(_, _key) for _ in metrics]

        # val list
        lst_v = []

        for _id in [m._id for m in metrics]:
            if _val == 'value':
                lst_v.append(self._hmp[_id][0].from_meta(self._hmp[_id][1]))

            elif _val == 'meta':
                lst_v.append(self._hmp[_id][1])

        return tuple(zip(lst_k, lst_v)) if _tuple else dict(zip(lst_k, lst_v))

    
    def is_better_than(self, tar, metrics):

        # calculate metrics
        hmp_0 = self.summary(metrics, _key='_id')
        hmp_1 =  tar.summary(metrics, _key='_id')

        for id_ in hmp_0:
            # compare
            rsl = self._hmp[id_][0].compare(hmp_0[id_], hmp_1[id_])
            
            if rsl == 1: 
                return True

            elif rsl == -1:
                return False
        
        return False


    def __repr__(self):

        str_ = ''

        for k, v in self.summary(_tuple=True):
            str_ += '{}: {}\n'.format(k, v.cpu().numpy())

        return str_