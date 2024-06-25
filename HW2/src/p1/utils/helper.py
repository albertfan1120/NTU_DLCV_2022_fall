import torch
import numpy as np
import os
from torchvision import transforms


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def get_device():
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(320)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    return device


def sample_noise(batch, noise_dim):
    noise = torch.randn(batch, noise_dim, 1, 1) 
    return noise


def generate_output(noise_dim, num, G, device, root, noise = None):
    num_images = num
    if noise == None:
        noise = sample_noise(num_images, noise_dim)
    noise = noise.to(device)

    G.eval()
    with torch.no_grad():
        fake_images = G(noise) * 0.5 + 0.5

    if not os.path.isdir(root):
        os.makedirs(root)
        
    for i in range(1, num_images+1):
        image = transforms.ToPILImage()(fake_images[i-1])
        image_path = os.path.join(root, str(i).zfill(4) + ".png")
        image.save(image_path)


def save_checkpoint(checkpoint_path, model):
    root = checkpoint_path.replace(checkpoint_path.split('/')[-1], "")
    if not os.path.isdir(root):
        os.makedirs(root)
        
    state = {'state_dict': model.state_dict()}
    torch.save(state, checkpoint_path)
    print('Model saved to %s\n\n' % checkpoint_path)
    
    
def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('Model loaded from %s' % checkpoint_path)