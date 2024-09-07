import random
import os.path as osp
import numpy as np
import torch
import torch_geometric

PATH_RAW = osp.join('data', 'raw_dir')
PATH_PROCESSED = osp.join('data', 'processed_dir')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:2')
else:
    DEVICE = torch.device('cpu')

SEED = 42

def set_seed_all(seed):
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    
    # Ensure that operations are deterministic in PyTorch
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # PyTorch Geometric random seed
    torch_geometric.seed_everything(seed)
