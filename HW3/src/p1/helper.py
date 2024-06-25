import os 
import torch
import numpy as np


def fix_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    return device


def process_path(path):
    path = os.path.expanduser(path)
    
    if not os.path.isabs(path):
        path = os.path.join('..', path)
    
    return path