"""
 configure DREAM
 
 @TODO: REFACTOR config class
"""
       
class Config(object):
    def __init__(self, config):   
        self.cuda = config['cuda'] # 是否使用GPU，bool
        self.clip = config['clip']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate'] # Initial Learning Rate
        self.log_interval = config['log_interval']
        
        self.none_idx = config['none_idx']
        self.pn_pair_num = config['pn_pair_num']
        self.basket_pool_type = config['basket_pool_type']
        self.rnn_type = config['rnn_type']
        self.rnn_layer_num = config['rnn_layers']
        self.dropout = config['dropout']
        self.num_product = config['num_product'] # 商品数目，用于定义Embedding Layer
        self.embedding_dim = config['embedding_dim'] # 商品表述维数， 用于定义Embedding Layer
        self.checkpoint_dir = config['checkpoint_dir']