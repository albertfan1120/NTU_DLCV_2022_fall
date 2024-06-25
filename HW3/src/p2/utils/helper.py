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


def save_checkpoint(checkpoint_path, model):
    root = checkpoint_path.replace(checkpoint_path.split('/')[-1], "")
    if not os.path.isdir(root):
        os.makedirs(root)
        
    state = {'state_dict': model.state_dict(),}
    torch.save(state, checkpoint_path)
    print('Model saved to %s\n\n' % checkpoint_path)
    
    
def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('Model loaded from %s' % checkpoint_path)


def get_key_padding_mask(captions):
    """
    key_padding_mask
    """
    key_padding_mask = torch.zeros(captions.size())
    key_padding_mask[captions == 0] = -torch.inf
    return key_padding_mask.bool()