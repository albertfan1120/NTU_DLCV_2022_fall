import os, glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms



class P2dataset(Dataset):
    def __init__(self, root, transform, mode = 'train', augment = False):
        self.trans = transform
        self.aug = augment
        self.mode = mode
        if mode == 'train':
            self.data = glob.glob(os.path.join(root, '*sat.jpg'))
        elif mode == 'test':
            self.data = glob.glob(os.path.join(root, '*jpg'))
        
                                            
    def __getitem__(self, index):
        img_path = self.data[index]
        if self.mode == 'train':
            mask_path = img_path.replace("sat.jpg", "mask.png")
            image, mask = self.trans(Image.open(img_path)), transforms.ToTensor()(Image.open(mask_path))
            mask = self.transform_mask(mask)
            
            if self.aug:
                image, mask = self.augment(image, mask)
            
            return image, mask
        
        elif self.mode == 'test':
            image = self.trans(Image.open(img_path))
            return image, img_path 


    def __len__(self):
        return len(self.data)
    
    
    def transform_mask(self, mask):
        _, H, W = mask.shape
        mask_label = torch.zeros((H, W), dtype = torch.long)
        
        mask = mask * 255
        mask = (mask >= 128).to(torch.long)
        mask = 4 * mask[0, :, :] + 2 * mask[1, :, :] + mask[2, :, :]  
        mask_label[mask == 3] = 0  # (Cyan: 011) Urban land 
        mask_label[mask == 6] = 1  # (Yellow: 110) Agriculture land 
        mask_label[mask == 5] = 2  # (Purple: 101) Rangeland 
        mask_label[mask == 2] = 3  # (Green: 010) Forest land 
        mask_label[mask == 1] = 4  # (Blue: 001) Water 
        mask_label[mask == 7] = 5  # (White: 111) Barren land 
        mask_label[mask == 0] = 6  # (Black: 000) Unknown 
    
        return mask_label
    
    
    def augment(self, img, mask):
        p_rotate = np.random.uniform()
        p_flip = np.random.uniform()
        
        if p_flip >= 0.5: 
            img, mask = torch.stack((img[0].T, img[1].T, img[2].T)) , mask.T 
        
        if 0 <= p_rotate < 0.25: 
            img, mask = torch.rot90(img, 1, dims = (1,2)), torch.rot90(mask, 1)
        elif 0.25 <= p_rotate < 0.5: 
            img, mask = torch.rot90(img, -1, dims = (1,2)), torch.rot90(mask, -1)
        elif 0.5 <= p_rotate < 0.75: 
            img, mask = torch.rot90(img, 2, dims = (1,2)), torch.rot90(mask, 2)
    
        return img, mask
    
    
if __name__ == '__main__':
    root = '/home/albert/DLCV/HW1/hw1_data/hw1_data/p2_data/train'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                mean=0.5,
                std=0.5),
    ])
    trainset = P2dataset(root, transform)
    image, mask = trainset[1]