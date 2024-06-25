from tqdm.auto import tqdm
from utils.validation import validation
from utils.helper import save_checkpoint


def train(model, optimizer, criterion, scheduler, config):
    print('#####################################')
    print('\n          Training start!!!          \n')
    print('#####################################')
    
    epoch = config['epoch']
    trainset_loader =  config['trainset_loader']
    device = next(model.parameters()).device

    model.train()  # Important: set training mode
    log_interval = 100
    best_score = 0
    for ep in range(1, epoch+1):
        iteration = 0
        tqbm_bar = tqdm(trainset_loader)
        for data, target in tqbm_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                tqbm_bar.set_description('Train Epoch:{}, loss: {:.6f}'.format(ep, loss.item()))
            iteration += 1
            
        scheduler.step()    
        score = validation(model, criterion, config)
        if score > best_score: 
            best_score = score 
            save_checkpoint(config['save_path'], model, optimizer)
        