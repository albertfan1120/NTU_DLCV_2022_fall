# official package
import os, argparse
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# my package
from dataset import P1dataset
from model import VGG16, Resnext101
from utils.helper import get_device
from utils.train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW1/hw1_data/hw1_data/p1_data/")
    parser.add_argument('--save_path', default = "./save_model/model_resnext.pth")
    args = parser.parse_args()
    
    config = {}
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-15, 15)),
        transforms.RandomGrayscale(p = 0.1),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])
    
    data_root = args.data_root
    trainset = P1dataset(root = os.path.join(data_root, 'train_50'), transform = transform_train)
    validset = P1dataset(root = os.path.join(data_root, 'val_50'), transform = transform_val)

    print('Numbers of images in trainset:', len(trainset)) # Should print 22500
    print('Numbers of images in validset:', len(validset)) # Should print 2500

    trainset_loader = DataLoader(trainset, batch_size=64, shuffle = True, num_workers = 12)
    validset_loader = DataLoader(validset, batch_size=64, shuffle = False, num_workers = 12)
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
    model = Resnext101(num_out = 50).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5)
    train(model, optimizer, criterion, scheduler, config)
    