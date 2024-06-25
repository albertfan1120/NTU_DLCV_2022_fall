# officail package 
import argparse, os
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
import torch.optim.lr_scheduler as lr_scheduler

# my package 
from utils.helper import get_device
from utils.train import train
from dataset import P1dataset
from model import Generator, Discriminator_W, Discriminator_DC


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW2/hw2_data/face/")
    parser.add_argument('--save_path', default = "./save_model/DCGAN_G.pth")
    args = parser.parse_args()
    data_root, save_path = args.data_root, args.save_path
   

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = 0.5, std = 0.5),
    ])

    trainset = P1dataset(root = os.path.join(data_root, 'train'), transform = transform)
    print('Numbers of images in trainset:', len(trainset)) # Should print 38464


    trainset_loader = DataLoader(trainset, batch_size = 64, shuffle = True, num_workers = 12)

    dataiter = iter(trainset_loader)
    images = dataiter.next()
    print('Image tensor in each batch:', images.shape, images.dtype)
 
    
    device = get_device()
    print('Device used:', device)

    noise_dim = 100
    G = Generator(noise_dim = noise_dim).to(device)
    D = Discriminator_DC().to(device)
    opt_G = optim.Adam(G.parameters(), lr = 2e-4, betas = (0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr = 2e-4, betas = (0.5, 0.9))
    sche_G = lr_scheduler.StepLR(opt_G, step_size = 50, gamma = 0.8)
    sche_D = lr_scheduler.StepLR(opt_D, step_size = 50, gamma = 0.8)


    config = {
        "epoch": 2000,
        "noise_dim": noise_dim,
        "device": device,
        "criterion": nn.BCELoss(),
        "trainset_loader": trainset_loader,
        "val_path": os.path.join(data_root, 'val'),
        "generator": G,
        "discriminator": D,
        "save_path": save_path,
        'output_path': './output'
    }
    train(config, opt_G, opt_D, sche_G, sche_D, mode = 'DCGAN')