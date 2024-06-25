import torch
from tqdm.auto import tqdm
from utils.helper import get_schedule, forward_diffusion, save_checkpoint


def train(config, optimizer, scheduler):
    print('#####################################')
    print('\n          Training start!!!          \n')
    print('#####################################')

    epoch = config['epoch']
    trainset_loader =  config['trainset_loader']
    device = config['device']
    criterion = config['criterion']
    model = config['DDPM']
    step = config['step']
    save_path = config['save_path']

    params = get_schedule(step)
     
    log_interval = 10
    for ep in range(1, epoch+1):
        iteration = 0
        model.train() 
        tqbm_bar = tqdm(trainset_loader)
        for imgs, labels in tqbm_bar:
            B = imgs.shape[0]
            t = torch.randint(0, step, (B,))

            noise_imgs, noise = forward_diffusion(imgs, t, params['sqrt_alphas_cumprod'], 
                                                  params['sqrt_one_minus_alphas_cumprod'])

            noise_imgs, noise = noise_imgs.to(device), noise.to(device)
            t, labels = t.to(device), labels.to(device)
            
            optimizer.zero_grad()
            noise_pred = model(noise_imgs, t, labels)
            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                tqbm_bar.set_description('Train Epoch:{}, loss: {:.6f}'.format(ep, loss.item()))
            iteration += 1
        
        scheduler.step() 
    save_checkpoint(save_path, model, optimizer)