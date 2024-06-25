import os, glob
from PIL import Image
from torch.utils.data import Dataset


class P1dataset(Dataset):
    def __init__(self, root, preprocess):
        self.process = preprocess
        self.data = glob.glob(os.path.join(root, '*.png'))
        
        self.data.sort()

                              
    def __getitem__(self, index):
        path = self.data[index]
        image = self.process(Image.open(path))
        file_name = os.path.basename(path)
            
        return image, file_name
     
        
    def __len__(self):
        return len(self.data)