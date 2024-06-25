from tqdm.auto import tqdm
from utils.helper import save_checkpoint, get_key_padding_mask
import torch.nn as nn


def train(config, optimizer, scheduler):
    epoch = config['epoch']
    trainset_loader =  config['trainset_loader']
    device = config['device']
    criterion = config['criterion']
    model = config['model']
    save_path = config['save_path']

    log_interval = 10
    for ep in range(1, epoch+1):
        iteration = 0
        model.train()       
        tqbm_bar = tqdm(trainset_loader)
        for imgs, captions in tqbm_bar:
            captions_in = captions[:, :-1]
            target = captions[:, 1:].to(device)
            imgs, captions_in = imgs.to(device), captions_in.to(device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(captions_in.size()[-1]).to(device)
            tgt_key_padding_mask = get_key_padding_mask(captions_in).to(device)
            
            optimizer.zero_grad()
            out = model(imgs, captions_in, tgt_mask, tgt_key_padding_mask)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                tqbm_bar.set_description('Train Epoch:{}, loss: {:.6f}'.format(ep, loss.item()))
            iteration += 1
        
        scheduler.step() 
        save_checkpoint(save_path, model)
    