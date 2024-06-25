# official pachage
import os, argparse 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# my pachage
from dataset import P3dataset
from utils.helper import get_device
from utils.train import train
from model import DANN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW2/hw2_data/digits/")
    parser.add_argument('--save_path', default = "./save_model/DANN_usps.pth")
    args = parser.parse_args()
    data_root = args.data_root
    save_path = args.save_path

    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.3),
        transforms.RandomGrayscale(),
        transforms.RandomAdjustSharpness(2),
        transforms.RandomPosterize(3),
        transforms.RandomRotation((-15, 15)),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=0.5, std=0.5),
    ])

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                mean=0.5, std=0.5),
    ])

    # usps,svhn, mnistm
    source, target = 'mnistm', 'usps'
    src_trainset = P3dataset(os.path.join(data_root, source), transform_train, mode = 'train')
    tar_trainset = P3dataset(os.path.join(data_root, target), transform_train, mode = 'train')
    tar_validset = P3dataset(os.path.join(data_root, target), transform_valid, mode = 'val')
    print('Numbers of images in source_trainset:', len(src_trainset)) 
    print('Numbers of images in target_trainset:', len(tar_trainset)) 
    print('Numbers of images in target_validset:', len(tar_validset)) 


    batch_size = 64
    src_train_loader = DataLoader(src_trainset, batch_size=batch_size, shuffle = True, num_workers = 12)
    tar_train_loader = DataLoader(tar_trainset, batch_size=batch_size, shuffle = True, num_workers = 12)
    tar_valid_loader = DataLoader(tar_validset, batch_size=batch_size, shuffle = False, num_workers = 12)

    dataiter = iter(src_train_loader)
    images, labels = dataiter.next()
    print('Source images tensor in each batch:', images.shape, images.dtype)
    print('Label tensor in each batch:', labels.shape, labels.dtype) 

    device = get_device()
    print('Device used:', device)

    model = DANN().to(device)

    config = {
        "epoch": 100,
        "device": device,
        "criterion": nn.CrossEntropyLoss(),
        "s_trainset_loader": src_train_loader,
        "t_trainset_loader": tar_train_loader,
        "t_validset_loader": tar_valid_loader,
        "DANN": model,
        "save_path": save_path,
    }
    optimizer = optimizer = optim.Adam(model.parameters(), lr = 2e-4, betas = (0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.7)
    train(config, optimizer, scheduler)