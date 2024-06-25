# official package
import argparse
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# my package
from dataset import P2dataset
from utils.helper import get_device, fix_seed
from utils.train import train
from model import VL_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW3/hw3_data/p2_data/")
    parser.add_argument('--token_root', default = "/home/albert/DLCV/HW3/hw3_data/caption_tokenizer.json")
    parser.add_argument('--save_path', default = "./save_model/VL_7.pth")
    args = parser.parse_args()
    data_root = args.data_root
    token_root = args.token_root
    save_path = args.save_path

    fix_seed(320)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness = [0.5, 1.3], 
                               contrast   = [0.8, 1.5], 
                               saturation = [0.2, 1.5]),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    trainset = P2dataset(data_root, transform, token_root, mode = 'train')
    print('Numbers of captions in trainset:', len(trainset))
    

    batch_size = 32
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle = True, num_workers = 12)
    
    dataiter = iter(train_loader)
    imgs, caption = dataiter.next()
    print('Images tensor in each batch:', imgs.shape, imgs.dtype)
    print('Caption tensor in each batch:', caption.shape, caption.dtype)

    device = get_device()
    print('Device used:', device) 

    model = VL_model().to(device)

    config = {
        "epoch": 100,
        "device": device,
        # ignore index of <PAD>
        "criterion": nn.CrossEntropyLoss(ignore_index = 0),
        "trainset_loader": train_loader,
        "model": model,
        "save_path": save_path,
    }
    optimizer = optim.Adam(model.parameters(), lr = 3e-5, weight_decay = 1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.8)
    train(config, optimizer, scheduler)