# official package
import os, argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
from byol_pytorch import BYOL

# my package
from dataset import P2dataset
from utils.helper import get_device
from utils.train_pretrain import train_pretrain


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW4/hw4_data/mini/train")
    parser.add_argument('--csv_root', default = "/home/albert/DLCV/HW4/hw4_data/mini/train.csv")
    parser.add_argument('--save_path', default = "./save_model/renet50_pretrain.pth")
    args = parser.parse_args()
    data_root = args.data_root
    csv_root = args.csv_root
    save_path = args.save_path 

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip() ,
        transforms.RandomAffine(10, translate = (0.1 , 0.1), scale = (0.9 , 1.1)),
        transforms.ColorJitter(brightness = 0.15, contrast = 0.15, saturation = 0.15, hue = 0.15), 

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pretrainset = P2dataset(data_root, csv_root, transform)
    print('Numbers of images in pretrainset:', len(pretrainset))

    pretrainset_loader = DataLoader(pretrainset, batch_size=16, shuffle = True, num_workers = 12)

    dataiter = iter(pretrainset_loader)
    images, labels = dataiter.next()

    print('Image tensor in each batch:', images.shape, images.dtype)
    print('Label tensor in each batch:', labels.shape, labels.dtype) 

    device = get_device()
    print('Device used:', device)

    model = models.resnet50(pretrained = False).to(device)
    learner = BYOL(
        model,
        image_size = 128,
        hidden_layer = 'avgpool'
    )

    config = {
        "epoch": 500,
        "device": device,
        "loader": pretrainset_loader,
        "model": model,
        "learner": learner,
        "save_path": save_path,
    }
    optimizer = optim.Adam(learner.parameters(), lr = 3e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.8)
    train_pretrain(config, optimizer, scheduler)