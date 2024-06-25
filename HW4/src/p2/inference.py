# official package
import os, argparse, json
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# my pachage
from dataset import P2dataset
from utils.helper import get_device, load_checkpoint, process_path
from model import Resnet_SSL


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_root', default = "/home/albert/DLCV/HW4/hw4_data/office/val.csv")
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW4/hw4_data/office/val/")
    parser.add_argument('--save_path', default = "./output_p2/pred.csv")
    args = parser.parse_args()

    csv_root = process_path(args.csv_root)
    data_root = process_path(args.data_root)
    save_path = process_path(args.save_path)

    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testset =trainset = P2dataset(data_root, csv_root, transform, mode = 'test')
    test_loader = DataLoader(testset, batch_size = 32, shuffle = False, num_workers = 4)
    device = get_device()

    finetune_root = 'save_model/renet50_finetune.pth'
    model = Resnet_SSL(65).to(device)
    load_checkpoint(finetune_root, model)

    label2word_root = 'index2word.json'
    with open(label2word_root) as f: label2word = json.load(f)
    
    print('#####################################')
    print('\n          Testing start!!!          \n')
    print('#####################################')
    model.eval()
    predList, nameList = [], []
    with torch.no_grad():
        for data, file_names in test_loader:
            data = data.to(device)
            output = model(data)
            pred_batch = output.max(1, keepdim=True)[1].squeeze()
            
            predList += [label2word[str(singleBatch.item())] for singleBatch in pred_batch]
            nameList += [singleBatch for singleBatch in file_names]
    
    csv_name = os.path.basename(save_path)
    csv_root = save_path.replace(csv_name, '')
   
    if not os.path.isdir(csv_root):
        os.makedirs(csv_root)
    
    csv_dic = {
        "id": range(len(nameList)),
        "filename": nameList,
        "label": predList
    }
    pred_df = pd.DataFrame(csv_dic)
    pred_df.to_csv(save_path, index=False)
            
    print('\n#####################################')
    print('\n          Testing complete!!!          \n')
    print('#####################################')