# -*- coding: utf-8 -*-
"""
    train DREAM
    
    @TODO
     - Optimizer Choosing
     - Hyper-parameter tuning
"""
import constants
from config import Config
from dream import DreamModel
from data import Dataset, BasketConstructor
from utils import batchify, repackage_hidden

import os
import pdb
import torch
import pickle
import random
import numpy as np
from time import time
from math import ceil
from sklearn.model_selection import train_test_split

# CUDA environtments
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=constans.GPUS

# Prepare input
bc = BasketConstructor(constants.RAW_DATA_DIR, constants.FEAT_DATA_DIR)
# Users' baskets
ub_basket =bc.get_baskets('prior', reconstruct = False)

if constants.REORDER: # For reorder prediction
    # Users' reordered baskets 
    ub_rbks = bc.get_baskets('prior', reconstruct = False, reordered = True)
    # User's item history
    ub_ihis = bc.get_item_history('prior', reconstruct = False)
    # Train test split
    train_ub, test_ub, train_rbks, test_rbks, train_ihis, test_ihis = train_test_split(ub_basket, ub_rbks, ub_ihis, test_size = 0.2)
    del ub_basket, ub_rbks, ub_ihis
    train_ub, test_ub = Dataset(train_ub, train_rbks, train_ihis), Dataset(test_ub, test_rbks, test_ihis)
    del train_rbks, test_rbks, train_ihis, test_ihis
 else:
    del ub_basket
    train_ub, test_ub = Dataset(train_ub), Dataset(test_ub)

# Model config
dr_config = Config(constants.DREAM_CONFIG)
dr_model = DreamModel(dr_config)
if dr_config.cuda:
    dr_model.cuda()


# Optimizer
optim = torch.optim.Adam(dr_model.parameters(), lr = dr_config.learning_rate)
# optim = torch.optim.Adadelta(dr_model.parameters())
# optim = torch.optim.SGD(dr_model.parameters(), lr=dr_config.learning_rate, momentum=0.9)

def reorder_bpr_loss(re_x, his_x, dynamic_user, item_embedding, config):
    '''
        loss function for reorder prediction
        re_x padded reorder baskets
        his_x padded history bought items
    '''
    nll = 0
    ub_seqs = []
    for u, h, du in zip(re_x, his_x, dynamic_user):
        du_p_product = torch.mm(du, item_embedding.t()) # shape: max_len, num_item
        nll_u = [] # nll for user
        for t, basket_t in enumerate(u):
            if basket_t[0] != 0:
                pos_idx = torch.cuda.LongTensor(basket_t) if config.cuda else torch.LongTensor(basket_t)                               
                # Sample negative products
                neg = [random.choice(h[t]) for _ in range(len(basket_t))] # replacement
                # neg = random.sample(range(1, config.num_product), len(basket_t)) # without replacement
                neg_idx = torch.cuda.LongTensor(neg) if config.cuda else torch.LongTensor(neg)
                # Score p(u, t, v > v')
                score = du_p_product[t - 1][pos_idx] - du_p_product[t - 1][neg_idx]
                # Average Negative log likelihood for basket_t
                nll_u.append(- torch.mean(torch.nn.LogSigmoid()(score)))
        nll += torch.mean(torch.cat(nll_u))
    return nll

def bpr_loss(x, dynamic_user, item_embedding, config):
    '''
        bayesian personalized ranking loss for implicit feedback
        parameters:
        - x: batch of users' baskets
        - dynamic_user: batch of users' dynamic representations
        - item_embedding: item_embedding matrix
        - config: model configuration
    '''
    nll = 0
    ub_seqs = []
    for u,du in zip(x, dynamic_user):
        du_p_product = torch.mm(du, item_embedding.t()) # shape: max_len, num_item
        nll_u = [] # nll for user
        for t, basket_t in enumerate(u):
            if basket_t[0] != 0 and t != 0:
                pos_idx = torch.cuda.LongTensor(basket_t) if config.cuda else torch.LongTensor(basket_t)
                # Sample negative products
                neg = [random.choice(range(1, config.num_product)) for _ in range(len(basket_t))] # replacement
                # neg = random.sample(range(1, config.num_product), len(basket_t)) # without replacement
                neg_idx = torch.cuda.LongTensor(neg) if config.cuda else torch.LongTensor(neg)
                # Score p(u, t, v > v')
                score = du_p_product[t - 1][pos_idx] - du_p_product[t - 1][neg_idx]
                #Average Negative log likelihood for basket_t
                nll_u.append(- torch.mean(torch.nn.LogSigmoid()(score)))
        nll += torch.mean(torch.cat(nll_u))
    return nll

def train_dream():
    dr_model.train() # turn on training mode for dropout
    dr_hidden = dr_model.init_hidden(dr_config.batch_size) 
    total_loss = 0
    start_time = time()
    num_batchs = ceil(len(train_ub) / dr_config.batch_size)     
    for i,x in enumerate(batchify(train_ub, dr_config.batch_size)):
        baskets, lens, _ = x
        dr_hidden = repackage_hidden(dr_hidden) # repackage hidden state for RNN
        dr_model.zero_grad() # optim.zero_grad()
        dynamic_user, _  = dr_model(baskets, lens, dr_hidden)
        loss = bpr_loss(baskets, dynamic_user, dr_model.encode.weight, dr_config)
        loss.backward()
        
        # Clip to avoid gradient exploding
        torch.nn.utils.clip_grad_norm(dr_model.parameters(), dr_config.clip) 

        # Parameter updating
        # manual SGD
        # for p in dr_model.parameters(): # Update parameters by -lr*grad
        #    p.data.add_(- dr_config.learning_rate, p.grad.data)
        # adam 
        optim.step()

        total_loss += loss.data
        
        # Logging
        if i % dr_config.log_interval == 0 and i > 0:
            elapsed = (time() - start_time) * 1000 / dr_config.log_interval
            cur_loss = total_loss[0] / dr_config.log_interval # turn tensor into float
            total_loss = 0
            start_time = time()
            print('[Training]| Epochs {:3d} | Batch {:5d} / {:5d} | ms/batch {:02.2f} | Loss {:05.2f} |'.format(epoch, i, num_batchs, elapsed, cur_loss))

def train_reorder_dream():
    dr_model.train() # turn on training mode for dropout
    dr_hidden = dr_model.init_hidden(dr_config.batch_size) 
    
    total_loss = 0
    start_time = time()
    num_batchs = ceil(len(train_ub) / dr_config.batch_size)     
    for i,x in enumerate(batchify(train_ub, dr_config.batch_size, is_reordered = True)):
        baskets, lens, ids, r_baskets, h_baskets = x
        dr_hidden = repackage_hidden(dr_hidden) # repackage hidden state for RNN
        dr_model.zero_grad() # optim.zero_grad()
        dynamic_user, _  = dr_model(baskets, lens, dr_hidden)
        loss = reorder_bpr_loss(r_baskets, h_baskets, dynamic_user, dr_model.encode.weight, dr_config)

        try:
            loss.backward()
        except RuntimeError: # for debugging
            print('caching')
            tmp = {'baskets':baskets, 'ids':ids, 'r_baskets':r_baskets, 'h_baskets':h_baskets,
                   'dynamic_user':dynamic_user, 'item_embedding':dr_model.encode.weight}
            print(baskets)
            print(ids)
            print(r_baskets)
            print(h_baskets)
            print(item_embedding.encode.weight)
            print(dynamic_user.data)
            with open('tmp.pkl', 'wb') as f:
                pickle.dump(tmp, f, pickle.HIGHEST_PROTOCOL)
            break

        # Clip to avoid gradient exploding
        torch.nn.utils.clip_grad_norm(dr_model.parameters(), dr_config.clip) 

        # Parameter updating
        # manual SGD
        # for p in dr_model.parameters(): # Update parameters by -lr*grad
        #    p.data.add_(- dr_config.learning_rate, p.grad.data)
        # adam 
        optim.step()

        total_loss += loss.data

        # Logging
        if i % dr_config.log_interval == 0 and i > 0:
            elapsed = (time() - start_time) * 1000 / dr_config.log_interval
            cur_loss = total_loss[0] / dr_config.log_interval # turn tensor into float
            total_loss = 0
            start_time = time()
            print('[Training]| Epochs {:3d} | Batch {:5d} / {:5d} | ms/batch {:02.2f} | Loss {:05.2f} |'.format(epoch, i, num_batchs, elapsed, cur_loss))

def evaluate_dream():
    dr_model.eval()
    dr_hidden = dr_model.init_hidden(dr_config.batch_size) 
    
    total_loss = 0
    start_time = time()
    num_batchs = ceil(len(test_ub) / dr_config.batch_size)
    for i,x in enumerate(batchify(test_ub, dr_config.batch_size)):
        baskets, lens, _ = x
        dynamic_user, _  = dr_model(baskets, lens, dr_hidden)
        loss = bpr_loss(baskets, dynamic_user, dr_model.encode.weight, dr_config)
        dr_hidden = repackage_hidden(dr_hidden)
        total_loss += loss.data
        
    # Logging
    elapsed = (time() - start_time) * 1000 / num_batchs
    total_loss = total_loss[0] / num_batchs
    print('[Evaluation]| Epochs {:3d} | Elapsed {:02.2f} | Loss {:05.2f} |'.format(epoch, elapsed, total_loss))
    return total_loss

def evaluate_reorder_dream():
    dr_model.eval()
    dr_hidden = dr_model.init_hidden(dr_config.batch_size) 

    total_loss = 0
    start_time = time()
    num_batchs = ceil(len(test_ub) / dr_config.batch_size)
    for i,x in enumerate(batchify(test_ub, dr_config.batch_size, is_reordered = True)):
        baskets, lens, _, r_baskets, h_baskets = x
        dynamic_user, _  = dr_model(baskets, lens, dr_hidden)
        loss = reorder_bpr_loss(r_baskets, h_baskets, dynamic_user, dr_model.encode.weight, dr_config)
        dr_hidden = repackage_hidden(dr_hidden)
        total_loss += loss.data

    # Logging
    elapsed = (time() - start_time) * 1000 / num_batchs
    total_loss = total_loss[0] / num_batchs
    print('[Evaluation]| Epochs {:3d} | Elapsed {:02.2f} | Loss {:05.2f} |'.format(epoch, elapsed, total_loss))
    return total_loss



best_val_loss = None

try:
    # print(dr_config)
    for k,v in constants.DREAM_CONFIG.items():
        print(k,v)
    # training
    for epoch in range(dr_config.epochs):
        if constants.REORDER:
            train_reorder_dream()
        else:
            train_dream()
        print('-' * 89)
        if constants.REORDER:
            val_loss = evaluate_reorder_dream()
        else:
            val_loss = evaluate_dream()
        print('-' * 89)
        # checkpoint
        if not best_val_loss or val_loss < best_val_loss:
            with open(dr_config.checkpoint_dir.format(epoch = epoch, loss = val_loss), 'wb') as f:
                torch.save(dr_model, f)
            best_val_loss = val_loss
        else:
            # Manual SGD slow down lr if no improvement in val_loss
            # dr_config.learning_rate = dr_config.learning_rate / 4
            pass
except KeyboardInterrupt:
    print('*' * 89)
    print('Early Stopping!')
