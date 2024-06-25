# official package
import os, argparse
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


# my package
from dataset import P2dataset
from model import Resnet_SSL
from utils.helper import get_device
from utils.train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW4/hw4_data/office")
    parser.add_argument('--save_path', default = "./save_model/renet50_finetune.pth")
    args = parser.parse_args()
    data_root = args.data_root
    save_path = args.save_path 

    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip() ,
        transforms.RandomAffine(10, translate = (0.1 , 0.1), scale = (0.9 , 1.1)),
        transforms.ColorJitter(brightness = 0.15, contrast = 0.15, saturation = 0.15, hue = 0.15), 

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = P2dataset(os.path.join(data_root, 'train'), os.path.join(data_root, 'train.csv'), transform_train)
    validset = P2dataset(os.path.join(data_root, 'val'), os.path.join(data_root, 'val.csv'), transform_val)
    print('Numbers of images in trainset:', len(trainset))
    print('Numbers of images in validset:', len(validset))

    trainset_loader = DataLoader(trainset, batch_size=32, shuffle = True, num_workers = 12)
    validset_loader = DataLoader(validset, batch_size=32, shuffle = False, num_workers = 12)
    
    dataiter = iter(trainset_loader)
    images, labels = dataiter.next()

    print('Image tensor in each batch:', images.shape, images.dtype)
    print('Label tensor in each batch:', labels.shape, labels.dtype) 

    device = get_device()
    print('Device used:', device)

    pretrained_root = 'save_model/renet50_pretrain.pth'
    model = Resnet_SSL(class_num = 65, pretrained_root = pretrained_root).to(device)

    config = {
        "epoch": 50,
        "device": device,
        "criterion": nn.CrossEntropyLoss(),
        "trainset_loader": trainset_loader,
        "validset_loader": validset_loader,
        "model": model,
        "save_path": save_path,
    }
    optimizer = optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.7)
    train(config, optimizer, scheduler)