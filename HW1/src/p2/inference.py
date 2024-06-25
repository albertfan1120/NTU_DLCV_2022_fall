# official package
import os, argparse
import numpy as np
import imageio
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# my package
from dataset import P2dataset
from utils.helper import get_device, load_checkpoint
from model import PSPNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW1/hw1_data/hw1_data/p2_data/my_test")
    parser.add_argument('--save_root', default = "output/pred_dir")
    args = parser.parse_args()
    config = {}
   
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = 0.5, 
            std = 0.5),
    ])
    
    if os.path.isabs(args.data_root):
        data_root = args.data_root
    else: 
        data_root = os.path.join('..', args.data_root)
    
    testset = P2dataset(root = data_root, transform = transform_test, mode = 'test') 
    testset_loader = DataLoader(testset, batch_size=8, shuffle = False, num_workers = 4)
    device = get_device()
    
    model = PSPNet(7).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 2e-4)
    load_checkpoint('./save_model/model_PSP.pth', model, optimizer)
    
    cls_color = {
        0:  [0, 255, 255],
        1:  [255, 255, 0],
        2:  [255, 0, 255],
        3:  [0, 255, 0],
        4:  [0, 0, 255],
        5:  [255, 255, 255],
        6: [0, 0, 0],
    }
    
    print('#####################################')
    print('\n          Testing start!!!          \n')
    print('#####################################')
    
    model.eval()
    predList, pathList = [], []
    with torch.no_grad():
        for image, imae_path in testset_loader:
            image = image.to(device)
            output = model(image)
            
            predList += [singleBatch for singleBatch in output.cpu().numpy()]
            pathList += [singleBatch for singleBatch in imae_path]
            
    pred = np.array(predList).argmax(axis=1) # (257, 512, 512)
    num_test = pred.shape[0]
    output_img = np.zeros((num_test, 512, 512, 3)) 
    for i in range(7):
        output_img[pred == i] = cls_color[i]
    
    if os.path.isabs(args.save_root):
        save_root = args.save_root
    else: 
        save_root = os.path.join('..', args.save_root)

    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    for i in range(num_test):
        imageio.imsave(os.path.join(save_root, os.path.basename(pathList[i]).replace('jpg', 'png')), 
                       np.uint8(output_img[i]))
        
    print('\n#####################################')
    print('\n          Testing complete!!!          \n')
    print('#####################################')