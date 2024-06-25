import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class P2dataset(Dataset):
    def __init__(self, data_root, csv_root, transform, mode = 'train'):
        self.data_root = data_root 
        self.trans = transform 
        self.mode = mode

        csv_file = pd.read_csv(csv_root)
        self.filename = list(csv_file.loc[:, 'filename'])
        if mode != 'test':
            label_name = list(csv_file.loc[:, 'label'])
            self.label = list(csv_file.loc[:, 'label'].astype('category').cat.codes)
            self.label_dict = dict(list(zip(self.label, label_name)))
        
                              
    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.filename[index])
        img = self.trans(Image.open(img_path).convert('RGB'))
        
        if self.mode != 'test':
            label = self.label[index]
            return img, label
        else: 
            filename = os.path.basename(img_path)
            return img, filename
     
        
    def __len__(self):
        return len(self.filename)


if __name__ == '__main__':
    from torchvision import transforms
    data_root = '/home/albert/DLCV/HW4/hw4_data/mini/train'
    csv_root = '/home/albert/DLCV/HW4/hw4_data/mini/train.csv'
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])

    dataset = P2dataset(data_root, csv_root, transform)
    print(len(dataset))

    img, label = dataset[0]
    print(img.shape)
    print(label)