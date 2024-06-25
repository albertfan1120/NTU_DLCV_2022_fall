# official package
import os, argparse
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# my package
from dataset import P1dataset
from utils.helper import get_device
from model import Resnext101
from utils.helper import load_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW1/hw1_data/hw1_data/p1_data/val_50")
    parser.add_argument('--save_path', default = "output/pred.csv")
    args = parser.parse_args()
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])
    
    if os.path.isabs(args.data_root):
        data_root = args.data_root
    else: 
        data_root = os.path.join('..', args.data_root)
        
    testset = P1dataset(root = data_root, transform = transform_test, mode = 'test') 
    testset_loader = DataLoader(testset, batch_size=32, shuffle = False, num_workers = 4)
    device = get_device()
    
    
    model = Resnext101(num_out = 50).to(device)
    optimizer = optim.SGD(model.parameters(), 1e-4)
    load_checkpoint('./save_model/model_resnext.pth', model, optimizer)
    
    print('#####################################')
    print('\n          Testing start!!!          \n')
    print('#####################################')
    model.eval()
    predList, nameList = [], []
    with torch.no_grad():
        for data, file_names in testset_loader:
            data = data.to(device)
            output = model(data)
            pred_batch = output.max(1, keepdim=True)[1].squeeze()
            
            predList += [int(singleBatch) for singleBatch in pred_batch]
            nameList += [singleBatch for singleBatch in file_names]
    
    if os.path.isabs(args.save_path):
        csv_path = args.save_path
    else: 
        csv_path = os.path.join('..', args.save_path)

    csv_name = os.path.basename(csv_path)
    csv_root = csv_path.replace(csv_name, '')
   
    if not os.path.isdir(csv_root):
        os.makedirs(csv_root)
    
    csv_dic = {
        "filename" : nameList,
        "label" : predList
    }
    pred_df = pd.DataFrame(csv_dic)
    pred_df.to_csv(csv_path, index=False)
            
    print('\n#####################################')
    print('\n          Testing complete!!!          \n')
    print('#####################################')