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

- The weights are initialized in the range uniform weight -0.1 and +0.1 uniformly. More insight can be gained at https://stackoverflow.com/a/55546528/13858953
  
#### init_hidden

- self.parameters() is a generator method that iterates over the parameters of the model. next retrieves the next item from the iterator by calling its next() method. Here, it returns the first parameter from the class.
- Now if the RNN type is LSTM or not we branch into 2 conditionals and then weight.new() creates a tensor that has the same data type, same device as the produced parameter. You can create a Variable in any fashion, but you need to specify the data type under such circumstance. For more insight read https://discuss.pytorch.org/t/what-does-next-self-parameters-data-mean/1458
- As far as I could understand zero_() is used to zero all gradients at the start of a minibatch. Also the parameters inside weight.new seem to indicate the dimensions. 
- Since the original code was written Variable has been deprecated. It still works as expected, but returns Tensors instead of Variables.

# train.py

## Important Imports - 
 
 - tensorboardX summary writer lets you log PyTorch models and metrics into a directory for visualization within the TensorBoard UI. Scalars, images, histograms, graphs, and embedding visualizations are all supported for PyTorch models and tensors as well as Caffe2 nets and blobs.

## Functions - 

### reorder_bpr_loss

- A matrix multiplication is performed between a entry from dynamic_user which we say earlier and the transposed item_embeddings. torch.mm does that.
- Next we iterate over a padded reorder basket. If it's first entry is non-zero we convert it to a tensor and use CUDA GPU if possible.
- Next we make a random list of size of basket and it's elements are chosen from the entries of the passed history bought items. This produces neg.
- neg_idx is again same as pos_idx
- We use the negative log likelihood to get score for a basket. We then use sigmoid over it and appent it to nll_u. nll likely stands for negative log likelihood.
- Once the inner loop runs out we add the average of the losses at each user to nll and return it

### bpr_loss

- Almost same as above
- Here neg is produced using random values between 1 and number of products

### train_dream

- We train the Dream model using this. 
- We build a hidden layer. 
- We then iterate, the batchify() function arranges the dataset into columns, trimming off any tokens remaining after the data has been divided into batches of size batch_size. For more insight read here - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
- In each iteration we find the bpr_loss and then compute it's gradient.
- The gradients are clipped as well based on the estabilished clip in the constants.
- get_grad_norm gets the norm of the gradients
- Any changes made to the copy formed using deep copy do not reflect in the original object. So we get the previous parameters.
- optim.step() updates the parameters
- We then compute the total loss, log the new paramters, update the weights and also compute the raio of norms of how much the weights changed by (delta) w.r.t norm of the parameters. They are implemented in the utils.py file.
- The logging function only executes at specific consitions and prints useful info

### train_reorder_dream

- Almost same as above. More debugging present and outputted as a pickle file if unable to run

### evaluate_dream

- We use dr.eval as model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you.
- We then compute the total loss in a pretty obvious way

### evaluate_reorder_dream

- Same as above

### Remaining Code 

- Some CUDA related tasks using environment variables from local system
- Using BasketConstructor from data.py followed by get_baskets we generate user baskets.
- If REORDER is set to True in constants.py we generate new baskets with reordered set to True else we don't eitherways we split the dataset into 2
- We then create the long used dr_model using the configs file and passing it to the DreamModel.
- We also move it to the GPu if possible
- We set Adam as our optimizer
- We instantiate a tensorboard writer object
- We set up a try except statement to break if interrupted using a keyboard.  Else the model trains and if the val_loss achieved in an iteration is better then old ones we checkpoint it.

# eval.py

## Important Imports - 

- The module pdb defines an interactive source code debugger for Python programs. 

## Functions - 

### eval_pred

- item_embedding are generated
- .eval() is called to stop dropout etc.
- You recreate the hidden layer using dr_model.init_hidden(dr_model.config.batch_size). The hidden state stores the internal state of the RNN from predictions made on previous tokens in the current sequence, this allows RNNs to understand context. The hidden state is determined by the output of the previous token. When you predict for the first token of any sequence, if you were to retain the hidden state from the previous sequence your model would perform as if the new sequence was a continuation of the old sequence which would give worse results. Instead for the first token you initialise an empty hidden state, which will then be filled with the model state and used for the second token. Think about it this way: if someone asked you to classify a sentence and handed you the US constitution (irrelevant information) vs. if someone gave you some background context about the sentence and then asked you to classify the sentence. The explanation is taken from - https://stackoverflow.com/a/55351254/13858953
- You then make some lists and begin iterating over User Baskets converted to batches
- The ub item contains baskets, lengths and user ids
- In the already instantiated Dream Model (dr_model) you pass the 3 entries
- All of these go into forward method
- Since we only need the dynamic user entry we ignore the rest
- We then repackage the hidden values using a function from utils.py to remove any associated history
- Then you do a nested iteration over the batch to compute user ids and their scores

### eval_batch

- Standard repeated stuff in the start
- If reordering was performed you get some additional entries while iterating otherwise you not so you get 2 branched conditions
- <u, p> score is computed inside the for loop for each user in the batch

### eval_up

- To compute the latest <u,p> score you first instantiate a dynamic user object and a embeddings object, multiply them and return the score.

### get_dynamic_u

- Using dr_model the dynamic_user is produced similar to eval_pred

### get_item_embedding

- This method produces the item embeddings based on the product id (pid)
- Depending on it's type you need to unsqueeze it or leave it as it is

### Remaining Code 

#### Notes method from data.py are heavily used

- A basket is formed using BasketConstructor
- ub_basket produces all the user baskets
- Dataset object is then produced from the user baskets
- Finally up stores all the products purchased by the user
- Rest of the code is self explanatory

# data.py

## Basket Constructor

### Constructor 

- Data input is converted to become an attribute of it's objects

### get_orders, get_orders_items, get_users_orders, get_users_products, get_items, get_baskets and get_item_history

- All are Dataset based and very easy to follow along with the doc string specifying their use

## Dataset

### Constructor

- Attributes are set using the user baskets etc.
- Depending on if reordered or not different branched conditionals are present

### __getitem__ and __len__ 

- Self explanatory

# Utils.py

## Functions

### Pad

- Pads only the first dimension and zero_() sets gradients to zero before back prop starts

### sort_batch_of_lists, pad_batch_of_lists

- Easy to understand from code and doc string

### batchify

- Takes the dataset and makes batches out of it
- Depending on if reordered or not we branch into 2 conditionals, in the first branch we pick up chunks of desired size using data[i * batch_size : (i + 1) * batch_size] in each iteration and sort and pad them, incase there are leftovers which can't form one batch then we add random values to it's end and form the last batch. In the second conditional we do pretty much the same but just with less parameters.

### pool_max and pool_avg

- return max and avg of tensors

### repackage_hidden

- Removes all previous history to make new variables

### get_grad_norm, get_loss, get_ratio_update, get_weight_update

- Pretty self explanatory names and code

### Remaining Code

- Seems to be for testing

# config.py

- Takes the config data and sets it as attributes for a config object

# constants.py

- Contains all the directories and configs

# Make Recommendation Using DREAM.ipynb

- Brings it all together into one code file