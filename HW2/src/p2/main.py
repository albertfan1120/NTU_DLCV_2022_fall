# official package
import argparse
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# my package
from dataset import P2dataset
from model import UNet
from utils.helper import get_device
from utils.train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW2/hw2_data/digits/mnistm/")
    parser.add_argument('--save_path', default = "./save_model/ddpm_400.pth")
    args = parser.parse_args()
    data_root = args.data_root
    save_path = args.save_path

    transform = transforms.Compose([
        transforms.Resize((36, 36)),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    trainset = P2dataset(root = data_root, transform = transform, mode = 'train')
    print('Numbers of images in trainset:', len(trainset)) 
    

    batch_size = 32
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle = True, num_workers = 12)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print('Source images tensor in each batch:', images.shape, images.dtype)
    print('Label tensor in each batch:', labels.shape, labels.dtype)

    device = get_device()
    print('Device used:', device) 

    step = 400
    model = UNet(T = step, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2],
                 num_res_blocks=2, dropout=0.1).to(device)
    
    config = {
        "epoch": 200,
        'step': step,
        "device": device,
        "criterion": nn.MSELoss(),
        "trainset_loader": train_loader,
        "DDPM": model,
        "save_path": save_path,
    }
    optimizer = optim.Adam(model.parameters(), lr = 2e-4, betas = (0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.8)
    train(config, optimizer, scheduler)