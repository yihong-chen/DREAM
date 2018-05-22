GPUS = "5,3,2,4"
REORDER = False
RAW_DATA_DIR = './tmp/raw/'
FEAT_DATA_DIR = './tmp/feat/'
DREAM_MODEL_DIR = './tmp/dream/'

DREAM_CONFIG = {'basket_pool_type': 'max', # 'avg'
                'rnn_layers': 2,  # 2, 3
                'rnn_type': 'RNN_RELU',  #'RNN_TANH',#'GRU',#'LSTM',# 'RNN_RELU',
                'dropout': 0.5,
                # 'num_product': 49688 + 1, # padding idx = 0
                'num_product': 49688 + 1 + 1, 
                # 49688 products, padding idx = 0, none idx = 49689, none idx indicates no products
                'none_idx': 49689,
                'embedding_dim': 128, # 128
                'cuda': True, # True,
                'clip': 200,  # 0.25
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001, # 0.0001
                'log_interval': 1, # num of batchs between two logging
                'checkpoint_dir': DREAM_MODEL_DIR + '\dream-{epoch:02d}-{loss:.4f}.model',
                }
