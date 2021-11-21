"""
Created on Thu Aug 26 15:42:38 2021

@author: cxue2
"""

import torch
from .core import _fit, _evl
import torch.multiprocessing as mp


class Model:

    def __init__(self, net):
        
        self.net = net.cpu()

    def fit(
        self, kwargs_ldr_trn, losses, optimizers, devices=('cpu',), 
        n_epochs=None, kwargs_ldr_vld=None, save_mode=0, save_dir=None,
        metrics_disp=[], metrics_crit=[],
        batch_fn_trn=None, batch_fn_vld=None
    ):
        ''' 
        The interface of training routine.
        
        Parameters
        ----------
        kwargs_ldr_trn : dict
            Argument dictionary for PyTorch DataLoader.
            
        losses : list
            List of PyTorch losses.
            
        optimizers : list
            List of PyTorch optimizers.
            
        devices : Iterable
            Indicate the device(s) to mount. It is a list of id(s)
            specifying all the device(s) to distribute training task. If
            more than 1 devices are given, multi-GPU mode is then implied. 
            torch.distributed and torch.nn.parallel module will be
            utilized. The default is ('cpu',).
            
        n_epochs : int | None
            # of epochs to train the model. If None, then the training
            process will run endlessly. The default is None.
            
        kwargs_ldr_vld : dict
            Argument dictionary for PyTorch DataLoader. Similar to 
            'kwargs_ldr_trn', yet this one is for validation dataset.
            Being None implies no validation dataset is present. The
            default is None.
            
        save_mode : int
            If mode is 0, the network will NOT be saved.
            If mode is 1, the network is saved at each epoch.
            If mode is 2, save is then dependent on criterions.
            Otherwise, throw error.
            
        save_dir : str
            Please refer to the description of Model._manage_save().
            
        metrics_disp : list[str]
            List of strings signifying the performance metrics to show
            on tqdm progress bar. The default is [].
            
        metrics_crit : list[str].
            List of strings signifying the performance metrics for model
            selection criterions. The default is [].
        '''

        kwargs = locals()
        kwargs.pop('self')

        # is training distributed or not
        is_distributed = len(devices) > 1

        # launch training in single-device or multi-GPU mode
        if is_distributed:

            queue = mp.get_context('spawn').SimpleQueue()
            event = mp.get_context('spawn').Event()
            args = tuple(kwargs.values())

            ctx = torch.multiprocessing.spawn(
                _fit._pipeline, args=(is_distributed, self.net, queue, event, *args),
                nprocs=len(devices), join=False,
            )

            # block and wait for the trained model being copied from queue
            state_dict = queue.get()
            event.set()
            self.net.load_state_dict(state_dict)

            # join context
            ctx.join()

        else:

            _fit._pipeline(0, is_distributed, self.net, None, None, **kwargs)
            self.net.to('cpu')

    def eval(
        self, kwargs_ldr, devices=('cpu',), metrics_disp=[], batch_fn=None
    ):

        kwargs = locals()
        kwargs.pop('self')

        # is training distributed or not
        is_distributed = len(devices) > 1

        # launch training in single-device or multi-GPU mode
        if is_distributed:

            args = tuple(kwargs.values())
            torch.multiprocessing.spawn(_evl._pipeline, args=(is_distributed, self.net, *args), nprocs=len(devices), join=True)

        else:

            _evl._pipeline(0, is_distributed, self.net, **kwargs)

    def save(self, path):
        ''' Save model. '''
        
        torch.save(self.net.state_dict(), path)  
    
    def load(self, path):
        ''' Load model. '''
        
        self.net.load_state_dict(torch.load(path))