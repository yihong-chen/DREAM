# -*- coding: utf-8 -*-
"""
    Useful Functions
"""
import torch
from torch.autograd import Variable

import numpy as np
################### PADDING & BATCH
def pad(tensor, length):
    '''
        pad 1st dim
        remain 0th 2nd dim
    '''
    #pdb.set_trace()
    return torch.cat([tensor, tensor.new(tensor.size(0), length - tensor.size(1), *tensor.size()[2:]).zero_()], 1)

def sort_batch_of_lists(batch_of_lists, lens, uids, rbks = None, ihis = None):
    '''
        sort batch of lists according to len(list)
        descending
    '''
    sorted_idx = [i[0] for i in sorted(enumerate(lens), key = lambda x : x[1], reverse = True)]
    uids = [uids[i] for i in sorted_idx]
    lens = [lens[i] for i in sorted_idx]
    batch_of_lists = [batch_of_lists[i] for i in sorted_idx]
    if rbks is not None and ihis is not None:
        rbks = [rbks[i] for i in sorted_idx]
        ihis = [ihis[i] for i in sorted_idx]
        return batch_of_lists, lens, uids, rbks, ihis
    else:
        return batch_of_lists, lens, uids

def pad_batch_of_lists(batch_of_lists, max_len):
    '''
        pad batch of lists
    '''
    padded = [l + [[0]] * (max_len - len(l)) for l in batch_of_lists]
    return padded
    
def batchify(data, batch_size, is_reordered = False):
    '''
        turn dataset into iterable batches
    '''
    if is_reordered is True:
        num_batch = len(data) // batch_size
        for i in range(num_batch):
            baskets, lens, uids, rbks, ihis = data[i * batch_size : (i + 1) * batch_size] # load
            baskets, lens, uids, rbks, ihis = sort_batch_of_lists(baskets, lens, uids, rbks, ihis) # sort, max_len = lens[0]
            padded_baskets = pad_batch_of_lists(baskets, lens[0]) # pad
            padded_rbks = pad_batch_of_lists(rbks, lens[0])
            padded_ihis = pad_batch_of_lists(ihis, lens[0])
            yield padded_baskets, lens, uids, padded_rbks, padded_ihis
        
        if len(data) % batch_size != 0:
            residual = [i for i in range(num_batch * batch_size, len(data))] + list(np.random.choice(len(data), batch_size - len(data) % batch_size))    
            print(len(residual))
            baskets, lens, uids, rbks, ihis = map(list, zip(*[data[idx] for idx in residual]))
            baskets, lens, uids, rbks, ihis = sort_batch_of_lists(baskets, lens, uids, rbks, ihis)
            padded_baskets = pad_batch_of_lists(baskets, lens[0])
            padded_rbks = pad_batch_of_lists(rbks, lens[0])
            padded_ihis = pad_batch_of_lists(ihis, lens[0])
            yield padded_baskets, lens, uids, padded_rbks, padded_ihis

    else: 
        num_batch = len(data) // batch_size
        for i in range(num_batch):
            baskets, lens, uids = data[i * batch_size : (i + 1) * batch_size] # load
            baskets, lens, uids = sort_batch_of_lists(baskets, lens, uids) # sort, max_len = lens[0]
            padded_baskets = pad_batch_of_lists(baskets, lens[0]) # pad
            yield padded_baskets, lens, uids
        
        if len(data) % batch_size != 0:
            residual = [i for i in range(num_batch * batch_size, len(data))] + list(np.random.choice(len(data), batch_size - len(data) % batch_size))    
            print(len(residual))
            baskets, lens, uids = map(list, zip(*[data[idx] for idx in residual]))
            baskets, lens, uids = sort_batch_of_lists(baskets, lens, uids)
            padded_baskets = pad_batch_of_lists(baskets, lens[0])
            yield padded_baskets, lens, uids

####################### Model Related

def pool_max(tensor, dim):
    return torch.max(tensor, dim)[0]

def pool_avg(tensor, dim):
    return torch.mean(tensor, dim)

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

if __name__ == "__main__":
    tensor = torch.Tensor([[[2, 2],
                            [3, 3]
                           ],
                           [[2, 2],
                            [3, 3]
                           ]
                          ])
    padded = pad(tensor, 3)
    print("Test padding size 10...")
    print('original tensor', tensor)
    print('padded tensor', padded)