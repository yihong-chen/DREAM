Note all Configs are defined in constants.py as a dictionary DREAM_CONFIG. They are later used with help of config.py which is imported throughout.

# dream.py

## Imports -

from torch.autograd import Variable: 
- torch.autograd provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.
- Variable is now deprecated according to PyTorch Docs, Variables are no longer necessary to use autograd with tensors. Autograd automatically supports Tensors with requires_grad set to True

## Classes - 

### Dream Model 

#### Init

- Extends nn.module class which is the base class for all neural network modules.
- Uses config from config.py
- torch.nn.Embedding is used to make a simple lookup table that stores embeddings of a fixed dictionary and size. So **config.num_product** indicates number of embeddings i.e. size of dictionary of embeddings and **config.embedding_dim** represents the size of each embedding vector. **padding_idx** is an optional argument which if specified, the entries at padding_idx do not contribute to the gradient; therefore, the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”.
- self.pool is a clever setup (maybe just for me xD), there is essentially a dict which is being indexed using config.basket_pool_type. In the constants.py file config.basket_pool_type is set to max.
- Setting up the RNN layers is as follows, if the layer is a LSTM or a GRU we enter the first condition getattr() is used to get the value of the named attribute of an object. So like in this case we would look in the torch.nn module to look for the exact attribute. Then it takes the standard arguments. We see both the input size and attributes in hidden layer are set to config.embedding_dim. They set the number of layers to config.rnn_layer_num. If batch_first is True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Also dropout is set to config.dropout from the data. The second condition is pretty much the same with the exception of you having the choice to specify your choice of non-linearity.
  
#### forward

- A nested for loop is in use, in each inner iteration, the basket is converted to a long tensor and then resized to (1, len(basket)), then if a CUDA enabled GPU is available we use it. Then the self.encode method defined in the constructor is used to generate embeddings. Then we use the chosen pooling mechanism. pool_avg and pool_max are defined in utils.py. They are essentially returning the max or average depending on the chosen one. Finally after inner iterations are done, the embedded baskets are concatenated and unsqueezed to obtain the desirable shape and appended into the ub_seqs list.
- After both loops are run ub_seqs is again reshaped and GPU is enabled if present
- Then the torch.nn.utils.rnn.pack_padded_sequence function is called which packs a Tensor containing padded sequences of variable length which holds the data and list of batch_sizes of a packed sequence. Lengths is provided by user and is list of sequence lengths of each batch element. Since batch_first is True the input is expected in B x T x * format where B is batch size, T is length of longest sequence and * handles rest of the dimensions.
- These are then passed in self.rnn along with the hidden parameter of the method. These are the the number of expected features in the input x and the number of features in the hidden state h. It returns an output which is tensor containing the output features and h_n which is a tensor containing hidden layer features. Since torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.
- torch.nn.utils.rnn.pad_packed_sequence Pads a packed batch of variable length sequences. It is an inverse operation to pack_padded_sequence(). dynamic_user is the returned tensor. _ is used to handle other returned values (maybe..) 
- Finally the dynamic_user tensor is returned alongsaide the hidden feature tensor (h_n).

#### init_weight

- The weights are initialized in the range uniform weight -0.1 and +0.1 uniformly. More insight can be gained (here)[https://stackoverflow.com/a/55546528/13858953]
  
#### init_hidden

- self.parameters() is a generator method that iterates over the parameters of the model. next retrieves the next item from the iterator by calling its next() method. Here, it returns the first parameter from the class.
- Now if the RNN type is LSTM or not we branch into 2 conditionals and then weight.new() creates a tensor that has the same data type, same device as the produced parameter. You can create a Variable in any fashion, but you need to specify the data type under such circumstance. For more insight read (here)[https://discuss.pytorch.org/t/what-does-next-self-parameters-data-mean/1458]
- As far as I could understand zero_() is used to zero all gradients at the start of a minibatch. Also the parameters inside weight.new seem to indicate the dimensions. 
- Since the original code was written Variable has been deprecated. It still works as expected, but returns Tensors instead of Variables.