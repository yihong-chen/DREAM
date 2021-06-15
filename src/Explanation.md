# dream.py

## Imports -

from torch.autograd import Variable: 
- torch.autograd provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.
- Variable is now deprecated according to PyTorch Docs, Variables are no longer necessary to use autograd with tensors. Autograd automatically supports Tensors with requires_grad set to True
