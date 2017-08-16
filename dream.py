# -*- coding: utf-8 -*-
"""
    DREAM: Dynamic Recurrent Network for Next Basket Prediction
    Based on PyTorch
    
    @TODO: 
        - Efficiency Variabel Usage
        - Other initilizations for item/product embedding, for example LDA
"""
import random
import numpy as np

import torch
from torch.autograd import Variable
from utils import pool_max, pool_avg

class DreamModel(torch.nn.Module):
    '''
       Input Data: b_1, ... b_i ..., b_t
                   b_i stands for user u's ith basket
                   b_i = [p_1,..p_j...,p_n]
                   p_j stands for the  jth product in user u's ith basket
    '''
    def __init__(self, config):
        super(DreamModel, self).__init__()
        # Model configuration
        self.config = config
        # Layer definitons
        self.encode = torch.nn.Embedding(config.num_product, 
                                         config.embedding_dim,
                                         padding_idx = 0) # Item embedding layer, 商品编码
        self.pool = {'avg':pool_avg, 'max':pool_max}[config.basket_pool_type] # Pooling of basket
        # RNN type specify
        if config.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(torch.nn, config.rnn_type)(config.embedding_dim, 
                                                          config.embedding_dim, 
                                                          config.rnn_layer_num, 
                                                          batch_first=True, 
                                                          dropout=config.dropout)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[config.rnn_type]
            self.rnn = torch.nn.RNN(config.embedding_dim, 
                                    config.embedding_dim, 
                                    config.rnn_layer_num, 
                                    nonlinearity=nonlinearity, 
                                    batch_first=True, 
                                    dropout=config.dropout)
    
    def forward(self, x, lengths, hidden):
        # Basket Encoding 
        ub_seqs = [] # users' basket sequence
        for user in x: # x shape (batch of user, time_step, indice of product) nested lists
            embed_baskets = []
            for basket in user:
                basket = torch.LongTensor(basket).resize_(1, len(basket))
                basket = basket.cuda() if self.config.cuda else basket # use cuda for acceleration
                basket = self.encode(torch.autograd.Variable(basket)) # shape: 1, len(basket), embedding_dim
                embed_baskets.append(self.pool(basket, dim = 1))
            # concat current user's all baskets and append it to users' basket sequence
            ub_seqs.append(torch.cat(embed_baskets, 1)) # shape: 1, num_basket, embedding_dim
        
        # Input for rnn 
        ub_seqs = torch.cat(ub_seqs, 0).cuda() if self.config.cuda else torch.cat(ub_seqs, 0) # shape: batch_size, max_len, embedding_dim
        packed_ub_seqs = torch.nn.utils.rnn.pack_padded_sequence(ub_seqs, lengths, batch_first=True) # packed sequence as required by pytorch
        
        # RNN
        output, h_u = self.rnn(packed_ub_seqs, hidden)
        dynamic_user, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # shape: batch_size, max_len, embedding_dim
        return dynamic_user, h_u
        
    def init_weight(self):
        # Init item embedding
        initrange = 0.1
        self.encode.weight.data.uniform_(-initrange, initrange)
    
    def init_hidden(self, batch_size):
        # Init hidden states for rnn
        weight = next(self.parameters()).data
        if self.config.rnn_type == 'LSTM':
            return (Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.embedding_dim).zero_()),
                    Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.embedding_dim).zero_()))
        else:
            return Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.embedding_dim).zero_())   
        