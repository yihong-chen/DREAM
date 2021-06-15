Note all Configs are defined in constants.py as a dictionary DREAM_CONFIG. They are later used with help of config.py which is imported throughout.

# dream.py

## Imports -

from torch.autograd import Variable: 
- torch.autograd provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.
- Variable is now deprecated according to PyTorch Docs, Variables are no longer necessary to use autograd with tensors. Autograd automatically supports Tensors with requires_grad set to True

## Classes - 

### Dream Model 

- Extends nn.module class which is the base class for all neural network modules.
- Uses config from config.py
- torch.nn.Embedding is used to make a simple lookup table that stores embeddings of a fixed dictionary and size. So *config.num_product* indicates number of embeddings i.e. size of dictionary of embeddings and *config.embedding_dim* represents the size of each embedding vector. padding_idx is an optional argument which if specified, the entries at padding_idx do not contribute to the gradient; therefore, the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”.
- 