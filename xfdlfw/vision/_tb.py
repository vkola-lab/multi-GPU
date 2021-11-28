"""
Created on Thu Aug 26 15:42:38 2021

@author: cxue2
"""

def tensorboard_fn_default(tb_writer, mode, device, net, ldr, rsl_epoch, epoch):

    for k, v in rsl_epoch.summary().items():
        tb_writer.add_scalar('{}/{}'.format(k, mode), v, epoch)