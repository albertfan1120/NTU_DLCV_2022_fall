from tqdm.auto import tqdm
from utils.helper import save_checkpoint


def train_pretrain(config, optimizer, scheduler):
    epoch = config['epoch']
    loader =  config['loader']
    device = config['device']
    model = config['model']
    learner = config['learner']
    save_path = config['save_path']

    log_interval = 10
    for ep in range(1, epoch+1):
        iteration = 0
        tqbm_bar = tqdm(loader)
        
        # don't use label in pretrain stage
        for imgs, _ in tqbm_bar:
            imgs = imgs.to(device)
    
            loss = learner(imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average() # update moving average of target encoder

            if iteration % log_interval == 0:
                tqbm_bar.set_description('Train Epoch:{}, loss: {:.6f}'.format(ep, loss.item()))
            iteration += 1
        
        scheduler.step() 
        save_checkpoint(save_path, model)