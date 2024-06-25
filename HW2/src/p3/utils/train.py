import torch
import numpy as np
from tqdm.auto import tqdm
from utils.validation import validation
from utils.helper import save_checkpoint


def train(config, optimizer, scheduler):
    print('#####################################')
    print('\n          Training start!!!          \n')
    print('#####################################')

    epoch = config['epoch']
    device = config['device']
    dann = config['DANN']
    save_path = config["save_path"]
    criterion = config['criterion']
    s_trainset_loader = config['s_trainset_loader']
    t_trainset_loader = config['t_trainset_loader']
    

    log_interval = 50
    best_score = 0
    for ep in range(1, epoch+1):
        dann.train()
        iteration = 0

        len_dataloader = min(len(s_trainset_loader), len(t_trainset_loader))
        tqbm_bar = tqdm(zip(s_trainset_loader, t_trainset_loader), total = len_dataloader)

        for step, ((src_imgs, src_labels), (tar_imgs, _)) in enumerate(tqbm_bar):
            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tar_imgs = tar_imgs.to(device)
            
            s_domain_label = torch.zeros(src_imgs.shape[0], dtype=torch.long).to(device)
            t_domain_label = torch.ones(tar_imgs.shape[0], dtype=torch.long).to(device)
            
            p = float(step + (ep-1) * len_dataloader) / (epoch * len_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # =============== start training ====================
            optimizer.zero_grad()
            
            # train on source data
            class_output, domain_output = dann(src_imgs, alpha)
            s_class_loss = criterion(class_output, src_labels)
            s_domain_loss = criterion(domain_output, s_domain_label)
            
            # train on target data
            _, domain_output = dann(tar_imgs, alpha)
            t_domain_loss = criterion(domain_output, t_domain_label)
            
            loss = s_class_loss + s_domain_loss + t_domain_loss
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                tqbm_bar.set_description('Train Epoch:{}, loss: {:.6f}'.format(ep, loss.item()))
            iteration += 1
            
        scheduler.step()
        score = validation(config)
        if score > best_score: 
            best_score = score 
            save_checkpoint(save_path, dann, optimizer)

