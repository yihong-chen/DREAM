# -*- coding: utf-8 -*-
"""
    Prepare Input for DREAM 
    Based on the Instartcart Dataset
"""
import os
import pickle
import numpy as np
import pandas as pd

import pdb

class BasketConstructor(object):
    '''
        Group products into baskets(type: list)
    '''
    def __init__(self, raw_data_dir, cache_dir):
        self.raw_data_dir = raw_data_dir
        self.cache_dir = cache_dir
    
    def get_orders(self):
        '''
            get order context information
        '''
        orders = pd.read_csv(self.raw_data_dir + 'orders.csv')
        orders = orders.fillna(0.0)
        orders['days'] = orders.groupby(['user_id'])['days_since_prior_order'].cumsum()
        orders['days_last'] = orders.groupby(['user_id'])['days'].transform(max)
        orders['days_up_to_last'] = orders['days_last'] - orders['days']
        del orders['days_last']
        del orders['days']
        return orders
    
    def get_orders_items(self, prior_or_train):
        '''
            get detailed information of prior or train orders 
        '''
        orders_products = pd.read_csv(self.raw_data_dir + 'order_products__%s.csv'%prior_or_train)
        return orders_products
    

    def get_users_orders(self, prior_or_train):
        '''
            get users' prior detailed orders
        '''
        if os.path.exists(self.cache_dir + 'users_orders.pkl'):
            with open(self.cache_dir + 'users_orders.pkl', 'rb') as f:
                users_orders = pickle.load(f)
        else:
            orders = self.get_orders()
            order_products_prior = self.get_orders_items(prior_or_train)
            users_orders = pd.merge(order_products_prior, orders[['user_id', 'order_id', 'order_number', 'days_up_to_last']], 
                        on = ['order_id'], how = 'left')
            with open(self.cache_dir + 'users_orders.pkl', 'wb') as f:
                pickle.dump(users_orders, f, pickle.HIGHEST_PROTOCOL)
        return users_orders
    
    def get_users_products(self, prior_or_train):
        '''
            get users' all purchased products
        '''
        if os.path.exists(self.cache_dir + 'users_products.pkl'):
            with open(self.cache_dir + 'users_products.pkl', 'rb') as f:
                users_products = pickle.load(f)
        else:
            users_products = self.get_users_orders(prior_or_train)[['user_id', 'product_id']].drop_duplicates()
            users_products['product_id'] = users_products.product_id.astype(int)
            users_products['user_id'] = users_products.user_id.astype(int)
            users_products = users_products.groupby(['user_id'])['product_id'].apply(list).reset_index()
            with open(self.cache_dir + 'users_products.pkl', 'wb') as f:
                pickle.dump(users_products, f, pickle.HIGHEST_PROTOCOL)
        return users_products

    def get_items(self, gran):
        '''
            get items' information
            gran = [departments, aisles, products]
        '''
        items = pd.read_csv(self.raw_data_dir + '%s.csv'%gran)
        return items
    
    def get_baskets(self, prior_or_train, reconstruct = False, reordered = False, none_idx = 49689):
        '''
            get users' baskets
        '''
        if reordered:
            filepath = self.cache_dir + './reorder_basket_' + prior_or_train + '.pkl'
        else:
            filepath = self.cache_dir + './basket_' + prior_or_train + '.pkl'
       
        if (not reconstruct) and os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                up_basket = pickle.load(f)
        else:          
            up = self.get_users_orders(prior_or_train).sort_values(['user_id', 'order_number', 'product_id'], ascending = True)
            uid_oid = up[['user_id', 'order_number']].drop_duplicates()
            up = up[up.reordered == 1][['user_id', 'order_number', 'product_id']] if reordered else up[['user_id', 'order_number', 'product_id']]
            up_basket = up.groupby(['user_id', 'order_number'])['product_id'].apply(list).reset_index()
            up_basket = pd.merge(uid_oid, up_basket, on = ['user_id', 'order_number'], how = 'left')
            for row in up_basket.loc[up_basket.product_id.isnull(), 'product_id'].index:
                up_basket.at[row, 'product_id'] = [none_idx]
            up_basket = up_basket.sort_values(['user_id', 'order_number'], ascending = True).groupby(['user_id'])['product_id'].apply(list).reset_index()
            up_basket.columns = ['user_id', 'reorder_basket'] if reordered else ['user_id', 'basket']
            #pdb.set_trace()
            with open(filepath, 'wb') as f:
                pickle.dump(up_basket, f, pickle.HIGHEST_PROTOCOL)
        return up_basket
        
    def get_item_history(self, prior_or_train, reconstruct = False, none_idx = 49689):
        filepath = self.cache_dir + './item_history_' + prior_or_train + '.pkl'
        if (not reconstruct) and os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                item_history = pickle.load(f)
        else:
            up = self.get_users_orders(prior_or_train).sort_values(['user_id', 'order_number', 'product_id'], ascending = True)
            item_history = up.groupby(['user_id', 'order_number'])['product_id'].apply(list).reset_index()
            item_history.loc[item_history.order_number == 1, 'product_id'] = item_history.loc[item_history.order_number == 1, 'product_id'] + [none_idx]
            item_history = item_history.sort_values(['user_id', 'order_number'], ascending = True)
            # accumulate 
            item_history['product_id'] = item_history.groupby(['user_id'])['product_id'].transform(pd.Series.cumsum)
            # get unique item list
            item_history['product_id'] = item_history['product_id'].apply(set).apply(list)
            item_history = item_history.sort_values(['user_id', 'order_number'], ascending = True)
            # shift each group to make it history
            item_history['product_id'] = item_history.groupby(['user_id'])['product_id'].shift(1)
            for row in item_history.loc[item_history.product_id.isnull(), 'product_id'].index:
                item_history.at[row, 'product_id'] = [none_idx]
            item_history = item_history.sort_values(['user_id', 'order_number'], ascending = True).groupby(['user_id'])['product_id'].apply(list).reset_index()
            item_history.columns = ['user_id', 'history_items']

            with open(filepath, 'wb') as f:
                pickle.dump(item_history, f, pickle.HIGHEST_PROTOCOL)
        return item_history 

class Dataset(object):
    '''
        Dataset prepare from user-basket
    '''
    def __init__(self, up_basket, up_r_basket = None, up_his = None):
        if (up_r_basket is not None) and (up_his is not None):
            self.is_reordered_included = True
        else:
            self.is_reordered_included = False

        up_basket['num_baskets'] = up_basket.basket.apply(len)
        self.user_id = list(up_basket.user_id)
        self.num_baskets = [int(n) for n in list(up_basket.num_baskets)]    
        self.basket = [[[int(p) for p in b]for b in u] for u in list(up_basket.basket)]

        if self.is_reordered_included is True:
            up_basket = pd.merge(up_basket, up_r_basket, on = ['user_id'], how = 'left')
            up_basket = pd.merge(up_basket, up_his, on = ['user_id'], how = 'left')
            self.reorder_basket = [[[int(p) for p in b]for b in u] for u in list(up_basket.reorder_basket)]
            self.history_item = [[[int(p) for p in b]for b in u] for u in list(up_basket.history_items)]

    def __getitem__(self, index):
        '''
            return baskets & num_baskets
        '''
        if self.is_reordered_included is True:
            return self.basket[index], self.num_baskets[index], self.user_id[index], self.reorder_basket[index], self.history_item[index]
        else:
            return self.basket[index], self.num_baskets[index], self.user_id[index]

    
    def __len__(self):
        return len(self.user_id)      