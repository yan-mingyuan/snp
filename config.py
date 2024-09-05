import os.path as osp
import torch

PATH_RAW = osp.join('data', 'raw_dir')
PATH_PROCESSED = osp.join('data', 'processed_dir')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

SEED = 42
BATCH_SIZE = 32