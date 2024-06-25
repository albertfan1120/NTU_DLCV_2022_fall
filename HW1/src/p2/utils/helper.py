import torch
import os 


def get_device():
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(320)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    return device


def save_checkpoint(checkpoint_path, model, optimizer):
    root = checkpoint_path.replace(checkpoint_path.split('/')[-1], "")
    if not os.path.isdir(root):
        os.makedirs(root)
        
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('Model saved to %s\n\n' % checkpoint_path)
    
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('Model loaded from %s' % checkpoint_path)