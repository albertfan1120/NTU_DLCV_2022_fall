import os, glob
from PIL import Image
from torch.utils.data import Dataset


class P1dataset(Dataset):
    def __init__(self, root, transform):
        self.trans = transform
        self.data = glob.glob(os.path.join(root, '*.png'))
        
           
    def __getitem__(self, index):
        img = self.trans(Image.open(self.data[index]))
        return img
        
       
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from torchvision import transforms
    root =  '/home/albert/DLCV/HW2/hw2_data/face/train'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = 0.5, std = 0.5),
    ])
    dataset = P1dataset(root, transform)
    img = dataset[0]
    print(img.shape)