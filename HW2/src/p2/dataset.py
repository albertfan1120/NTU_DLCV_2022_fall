import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class P2dataset(Dataset):
    def __init__(self, root, transform, mode):
        self.root = root 
        self.trans =  transform
        
        csv_file = pd.read_csv(os.path.join(root, mode + '.csv'))

        self.img_list = list(csv_file.loc[:, 'image_name'])
        self.label_list = list(csv_file.loc[:, 'label'])
        
                                  
    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'data', self.img_list[index])
        img = self.trans(Image.open(img_path).convert('RGB'))

        label = self.label_list[index]

        return img, label
        
       
    def __len__(self):
       return len(self.img_list)


if __name__ == '__main__':
    from torchvision import transforms
    root = '/home/albert/DLCV/HW2/hw2_data/digits/mnistm/'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    dataset = P2dataset(root, transform, 'train')
    img, label = dataset[0]
    print(img.shape)
    print(label)