# -*- coding: utf-8 -*- #
 
# ------------------------------------------------------------------
# File Name:        utils
# Author:           郑徐瀚宇
# Version:          ver0_1
# Created:          2023/11/21
# Description:      Miscellaneous tools
# Function List:    print_log() -- Output traning logs to path@./log
#                   np_append() -- Append Numpy type data like List type
#                   ts_append() -- Append Tensor type data like List type
# History:
#       <author>        <version>       <time>      <desc>
#       郑徐瀚宇         ver0_1          2023/11/21  None
# ------------------------------------------------------------------

import os
import numpy as np
import torch

norm_mean = 0
norm_std = 0

def print_log(i, args, results):
    """Print Training Config and Results
    ---
    Parameters:
        args: Training config
        results: Training results
    """

    f = open('./log/log' + args.model + '.txt', 'a')
    f.write(f"Trained EXP{i}:\n")
    f.write('Input Parameters:\n')
    
    for k, v in vars(args).items():
        f.write('\t'+str(k)+'='+str(v)+'\n')
    f.write('\n')
    f.write('Training Results:\n')
    for k, v in results.items():
        f.write('\t'+str(k)+': '+str(v)+'\n')
    f.write('\n')
    f.close()

def np_append(a, b):
    """List like 'Append' tool for numpy datatype
    ---
    Parameters:
        a, b: append a with b
    """
    shape = list(b.shape)
    shape.insert(0, -1)
    if a is None:
        a = np.array([]).reshape(tuple(shape))
    return np.append(a,b.reshape(tuple(shape)), axis = 0)

def ts_append(a, b):
    """List like 'Append' tool for tensor datatype
    ---
    Parameters:
        a, b: append a with b
    """
    if a is None:
        return b.unsqueeze(0)
    else:
        return torch.cat([a, b.unsqueeze(0)], dim=0)

def lb_normalize(labels):
    """ Normalize Labels
    """
    global norm_mean
    norm_mean = np.mean(labels)
    global norm_std
    norm_std = np.std(labels)

    normed_labels = (labels - norm_mean) / norm_std
    return normed_labels

def lb_denormalize(labels):
    """ Denormalize Labels
    """
    denormed_labels = labels*norm_std + norm_mean
    return denormed_labels