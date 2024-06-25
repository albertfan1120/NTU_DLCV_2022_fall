# official package
import os, argparse
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# my package
from dataset import P2dataset
from utils.helper import get_device
from utils.train import train
from model import VGG16_FCN32, PSPNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW1/hw1_data/hw1_data/")
    parser.add_argument('--save_path', default = "./save_model/model_PSP.pth")
    args = parser.parse_args()
    
    config = {}
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = 0.5, std = 0.5),
    ])
    
    data_root = args.data_root
    trainset = P2dataset(root = os.path.join(data_root, 'p2_data', 'train'), transform = transform, augment = True)
    validset = P2dataset(root = os.path.join(data_root, 'p2_data', 'validation'), transform = transform)

    print('Numbers of images in trainset:', len(trainset)) # Should print 2000
    print('Numbers of images in validset:', len(validset)) # Should print 257
    
    trainset_loader = DataLoader(trainset, batch_size=8, shuffle = True, num_workers = 12)
    validset_loader = DataLoader(validset, batch_size=8, shuffle = False, num_workers = 12)
    config['trainset_loader'] = trainset_loader
    config['validset_loader'] = validset_loader


    dataiter = iter(trainset_loader)
    images, labels = dataiter.next()

    print('Image tensor in each batch:', images.shape, images.dtype)
    print('Label tensor in each batch:', labels.shape, labels.dtype) 
    
    
    device = get_device()
    print('Device used:', device)
    
    config['epoch'] = 100
    config['save_path'] = args.save_path
    model = PSPNet(7).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.7)
    criterion = nn.CrossEntropyLoss()
    train(model, optimizer, criterion, scheduler, config)
    