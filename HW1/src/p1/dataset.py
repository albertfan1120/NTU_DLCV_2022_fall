import os, glob
from PIL import Image
from torch.utils.data import Dataset


class P1dataset(Dataset):
    def __init__(self, root, transform, mode = 'train'):
        self.trans = transform
        self.mode = mode
        self.data = glob.glob(os.path.join(root, '*.png'))
        
        self.data.sort()

                              
    def __getitem__(self, index):
        path = self.data[index]
        image = self.trans(Image.open(path))
        
        if self.mode == 'train':
            label, _ = os.path.basename(path).split('_')
            label = int(label)
            
            return image, label
        
        elif self.mode == 'test':
            file_name = os.path.basename(path)
            
            return image, file_name
     
        
    def __len__(self):
        return len(self.data)
    


if __name__ == '__main__':
    from torchvision import transforms
    root = '/home/albert/DLCV/HW1/hw1_data/hw1_data/p1_data/train_50'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])
    
    trainset = P1dataset(root, transform)
    image, label = trainset[0]
