import os, json
from PIL import Image
from torch.utils.data import Dataset
from tokenizers import Tokenizer
import torch, glob


class P2dataset(Dataset):
    def __init__(self, data_root, transform, token_root = None, mode = 'train'):
        self.root = data_root
        self.trans = transform
        self.mode = mode
        self.max_len = 54
        if mode == 'train':
            with open(os.path.join(data_root, mode + '.json')) as f: 
                token_dict = json.load(f)

            self.captions = token_dict['annotations']
            self.imgs = {img_dict['id']: img_dict['file_name'] for img_dict in token_dict['images']}
            self.tokenizer = Tokenizer.from_file(token_root)
        else:
            self.data = glob.glob(os.path.join(data_root, '*.jpg'))
            self.data.sort()


    def __getitem__(self, index):
        if self.mode == 'train':
            caption_info = self.captions[index]

            # img
            img_name = self.imgs[caption_info['image_id']]
            img_path = os.path.join(self.root, 'images', self.mode, img_name)
            img = self.trans(Image.open(img_path).convert('RGB'))
            
            # caption
            tokenized_caption = self.tokenizer.encode(caption_info['caption'])
            caption_ids = tokenized_caption.ids
            caplen = len(caption_ids)
            
            # padding with <pad>
            caption_ids += [0 for _ in range(self.max_len - caplen)]
            caption_ids = torch.Tensor(caption_ids).long()

            return img, caption_ids
        else:
            img_path =self.data[index]
            file_name = os.path.basename(img_path).split('.')[0]
            img = self.trans(Image.open(img_path).convert('RGB'))
            return img, file_name
        
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.captions)
        else:
            return len(self.data)




if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])
    root = '/home/albert/DLCV/HW3/hw3_data/p2_data/images/val'
    token_root = '/home/albert/DLCV/HW3/hw3_data/caption_tokenizer.json'
    dataset = P2dataset(root, token_root, transform, mode = 'train')