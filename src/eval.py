"""
    Evaluation of DREAM
"""
import torch
from math import ceil
from time import time

import constants
from config import Config
from data import Dataset, BasketConstructor
from utils import repackage_hidden, batchify

import pdb

def eval_pred(dr_model, ub):
    '''
        evaluate dream model for predicting next basket on all training users
        in batches
    '''
    item_embedding = dr_model.encode.weight
    dr_model.eval()
    dr_hidden = dr_model.init_hidden(dr_model.config.batch_size)
    start_time = time()
    id_u, score_u = [], [] # user's id, user's score
    num_batchs = ceil(len(ub) / dr_model.config.batch_size)
    for i,x in enumerate(batchify(ub, dr_model.config.batch_size)):
        print(i)
        baskets, lens, uids = x
        _, dynamic_user, _ = dr_model(baskets, lens, dr_hidden)# shape: batch_size, max_len, embedding_size
        dr_hidden = repackage_hidden(dr_hidden)
        for i,l,du in zip(uids, lens, dynamic_user):
            du_latest = du[l - 1].unsqueeze(0) # shape: 1, embedding_size
            score_up = torch.mm(du_latest, item_embedding.t()) # shape: 1, num_item
            score_u.append(score_up.cpu().data.numpy())
            id_u.append(i)
    elapsed = time() - start_time 
    print('[Predicting] Elapsed: {02.2f}'.format(elapsed))
    return score_ub, id_u

def eval_batch(dr_model, ub, up, batch_size, is_reordered = False):
    '''
        Using dr_model to predict (u,p) score in batch
        Parameters:
        - ub: users' baskets
        - up: users' history purchases
        - batch_size
    '''
    # turn on evaluation mode
    dr_model.eval()
    is_cuda =  dr_model.config.cuda
    item_embedding = dr_model.encode.weight
    dr_hidden = dr_model.init_hidden(batch_size)

    id_u, item_u, score_u, dynam_u = [], [], [], []
    start_time = time()
    num_batchs = ceil(len(ub) / batch_size)
    for i, x in enumerate(batchify(ub, batch_size, is_reordered)):
        if is_reordered is True:
            baskets, lens, uids, r_baskets, h_baskets = x
        else:
            baskets, lens, uids = x
        dynamic_user, _ = dr_model(baskets, lens, dr_hidden)
        for uid, l, du in zip(uids, lens, dynamic_user):
            du_latest =  du[l - 1].unsqueeze(0)
            # calculating <u,p> score for all history <u,p> pair 
            history_item = [int(i) for i in up[up.user_id == uid]['product_id'].values[0]]
            history_item = torch.cuda.LongTensor(history_item) if is_cuda else torch.LongTensor(history_item)
            score_up = torch.mm(du_latest, item_embedding[history_item].t()).cpu().data.numpy()[0]
            id_u.append(uid), dynam_u.append(du_latest.cpu().data.numpy()[0]), item_u.append(history_item.cpu().numpy()),score_u.append(score_up)
        # Logging
        elapsed = time() - start_time; start_time = time()
        print('[Predicting]| Batch {:5d} / {:5d} | seconds/batch {:02.02f}'.format(i, num_batchs, elapsed))
    return id_u, item_u, score_u, dynam_u


def eval_up(uid, pid, dr_model, ub, dr_hidden):
    '''
        calculate latest score for (u,p) pair
        ub for single user
        dr_hidden: should be initialized and repackage after each batch
    '''
    dynamic_user = get_dynamic_u(uid, dr_model, ub, dr_hidden)
    item_embedding = get_item_embedding(pid, dr_model)
    score_up = torch.mm(dynamic_user, item_embedding.t())
    return score_up

def get_dynamic_u(uid, dr_model, ub, dr_hidden):
    '''
        get latest dynamic representation of user uid
        dr_hidden must be provided as global variable
    '''
    for i,x in enumerate(batchify(ub, 1)):
        baskets, lens, uids = x
        _, dynamic_user, _ = dr_model(baskets, lens, dr_hidden)
    return dynamic_user[0][lens[0] - 1].unsqueeze(0)

def get_item_embedding(pid, dr_model):
    '''
        get item's embedding
        pid can be a integer or a torch.cuda.LongTensor
    '''
    if isinstance(pid, torch.cuda.LongTensor) or isinstance(pid, torch.LongTensor):
        return dr_model.encode.weight[pid]
    elif isinstance(pid, int):
        return dr_model.encode.weight[pid].unsqueeze(0)
    else:
        print('Unsupported Index Type %s'%type(pid))
        return None

if __name__ == '__main__':
    bc = BasketConstructor(constants.RAW_DATA_DIR, constants.FEAT_DATA_DIR)
    ub_basket = bc.get_baskets('prior', reconstruct = False)
    ub = Dataset(ub_basket)
    up = bc.get_users_products('prior')
    
    dr_config = Config(constants.DREAM_CONFIG)
    with open(dr_config.checkpoint_dir, 'rb') as f:
        dr_model = torch.load(f)
    dr_model.eval()

    # score_ub, id_u = eval_pred(dr_model, ub)
    # (u,p) score
    # uid, pid = 1, 196
    # ub = Dataset(ub_basket[ub_basket.user_id == 1])
    # dr_hidden = dr_model.init_hidden(1) # init hidden for batch_size = 1 prediction
    # score_up = eval_up(uid, pid, dr_model, ub, dr_hidden)
    # dr_hidden = repackage_hidden(dr_hidden)
    # print(score_up)

    # uid, pid = 1, torch.cuda.LongTensor([2, 4, 5])
    # ub = Dataset(ub_basket[ub_basket.user_id == 1])
    # dr_hidden = dr_model.init_hidden(1) # init hidden for batch_size = 1 prediction
    # score_up = eval_up(uid, pid, dr_model, ub, dr_hidden)
    # dr_hidden = repackage_hidden(dr_hidden)
    # print(score_up)
    
    ub = Dataset(ub_basket.head(64))
    
    id_u, item_u, score_u = eval_batch(dr_model, ub, up, 32)
    
    