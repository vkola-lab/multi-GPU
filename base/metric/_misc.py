#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

import torch
import functools
import inspect


def _numpy(fnc):

    @functools.wraps(fnc)
    def main(*args, **kwargs):

        # parse inputs
        kwargs_ = _parse_inputs(fnc, args, kwargs)

        # record device
        dev = kwargs_['output'].device

        # convert output and y_true to numpy arrays
        for k in ['output', 'y_true']: kwargs_[k] = kwargs_[k].detach().cpu().numpy()

        # call wrapped function and covert result to torch.Tensor
        return torch.tensor(fnc(**kwargs_), device=dev)
    
    return main


def _detach(fnc):

    @functools.wraps(fnc)
    def main(*args, **kwargs):

        # parse inputs
        kwargs_ = _parse_inputs(fnc, args, kwargs)

        # detach output and y_true
        for k in ['output', 'y_true']:
            
            kwargs_[k] = kwargs_[k].detach()

        return fnc(**kwargs_)
    
    return main


def _fn_tpl_compare_(fnc, sgn):

    @functools.wraps(fnc)
    def main(*args, **kwargs):

        # parse inputs
        kwargs_ = _parse_inputs(fnc, args, kwargs)

        # compare
        val_0, val_1 = kwargs_['val_0'], kwargs_['val_1']
        
        if val_0 > val_1:
            
            return 1 * sgn

        elif val_0 == val_1:
            
            return 0 * sgn

        else:
            
            return -1 * sgn
    
    return main

_fn_tpl_compare = lambda _: functools.partial(_fn_tpl_compare_, sgn=_)


def _parse_inputs(fnc, args, kwargs):

    sig = inspect.signature(fnc)
    sig = sig.bind(*args, **kwargs)
    sig.apply_defaults()
    kwargs_ = sig.arguments

    return kwargs_

