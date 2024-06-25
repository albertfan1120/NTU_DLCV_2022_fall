import os 
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
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


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_schedule(T):
    betas = linear_beta_schedule(timesteps = T)

    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    params = {
        'betas': betas,
        'sqrt_recip_alphas':  sqrt_recip_alphas,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod, 
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_variance': posterior_variance,
    }

    return params


def forward_diffusion(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        x_0 -> (B, C, H, W)
        t -> (B)
    """
    noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t][:, None, None, None] # (B, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None] # (B, 1, 1, 1)
    
    noise_imgs = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    return noise_imgs, noise


@torch.no_grad()
def sample_timestep(model, x, t, labels, params):
    model.eval()

    now_t = t[0] # all element in t are same, but t is 1D tensor
    betas_t = params['betas'][now_t]
    sqrt_one_minus_alphas_cumprod_t =  params['sqrt_one_minus_alphas_cumprod'][now_t]
    sqrt_recip_alphas_t = params['sqrt_recip_alphas'][now_t]
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, labels) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = params['posterior_variance'][now_t]
    
    if now_t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 




@torch.no_grad()
def generate_imgs(model, total_step, device, params, num_class = 10, num_per_class = 100):
    shape = (3, 32, 32)

    num_imgs = num_class * num_per_class
    noise = torch.randn((num_imgs,) + shape) # (1000, 3, 32, 32)
    
    output = torch.zeros((0,) + shape)
    for num in range(num_class):
        print('Process class {}'.format(num))

        labels = torch.full((num_per_class,), num).long()

        batch_imgs = noise[num_per_class*num:num_per_class*num+num_per_class]
        for time in tqdm(reversed( range(0, total_step) ), total = total_step):
            t = torch.full((num_per_class,), time).long()

            batch_imgs, labels, t = batch_imgs.to(device), labels.to(device), t.to(device)
            batch_imgs = sample_timestep(model, batch_imgs, t, labels, params)

        output = torch.cat((output, batch_imgs.cpu()), dim = 0)

    return output
        