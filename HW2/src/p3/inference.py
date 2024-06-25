# official package
import os, argparse
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# my pachage
from dataset import P3dataset
from utils.helper import get_device, load_checkpoint
from model import DANN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW2/hw2_data/digits/usps/data")
    parser.add_argument('--save_path', default = "./output_p3/pred.csv")
    args = parser.parse_args()

    data_root = os.path.expanduser(args.data_root)
    if not os.path.isabs(data_root):
        save_path = os.path.join('..', data_root)

    save_path = os.path.expanduser(args.save_path)
    if not os.path.isabs(save_path):
        save_path = os.path.join('..', save_path)

    if 'svhn' in data_root: 
        target = 'svhn'
    elif 'usps' in data_root: 
        target = 'usps'
    else: 
        target = None 

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    testset = P3dataset(data_root, transform, mode = 'test')
    test_loader = DataLoader(testset, batch_size=64, shuffle = False, num_workers = 4)
    device = get_device()

    model = DANN().to(device)
    optimizer = optimizer = optim.Adam(model.parameters(), lr = 2e-4, betas = (0.9, 0.999))
    model_path = os.path.join('save_model', 'DANN_' + target + '.pth')
    load_checkpoint(model_path, model, optimizer)

    print('#####################################')
    print('\n          Testing start!!!          \n')
    print('#####################################')
    model.eval()
    predList, nameList = [], []
    with torch.no_grad():
        for data, file_names in test_loader:
            data = data.to(device)
            output, _ = model(data)
            pred_batch = output.max(1, keepdim=True)[1].squeeze()
            
            predList += [int(singleBatch) for singleBatch in pred_batch]
            nameList += [singleBatch for singleBatch in file_names]
    

    csv_name = os.path.basename(save_path)
    csv_root = save_path.replace(csv_name, '')
   
    if not os.path.isdir(csv_root):
        os.makedirs(csv_root)
    
    csv_dic = {
        "image_name" : nameList,
        "label" : predList
    }
    pred_df = pd.DataFrame(csv_dic)
    pred_df.to_csv(save_path, index=False)
            
    print('\n#####################################')
    print('\n          Testing complete!!!          \n')
    print('#####################################')