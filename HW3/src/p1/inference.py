# official package
import os, argparse
import pandas as pd
import torch, json
import clip
from torch.utils.data import DataLoader

# my package
from dataset import P1dataset
from helper import fix_seed, get_device, process_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW3/hw3_data/p1_data/val")
    parser.add_argument('--label_root', default = "/home/albert/DLCV/HW3/hw3_data/p1_data/id2label.json")
    parser.add_argument('--save_path', default = "output_p1/pred.csv")
    args = parser.parse_args()

    data_root, label_root = process_path(args.data_root), process_path(args.label_root)
    save_path = process_path(args.save_path)

    fix_seed(320)
    device = get_device()
    model, preprocess = clip.load('ViT-B/32', device)
    with open(label_root) as f: label_dict = json.load(f)

    dataset = P1dataset(data_root, preprocess)
    data_loader = DataLoader(dataset, batch_size = 32, shuffle = False, num_workers = 12)
    
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in label_dict.values()])
    
    print('#####################################')
    print('\n          Testing start!!!          \n')
    print('#####################################')

    model.eval()
    nameList = []
    pred_feature = torch.empty((0, 512), dtype=torch.float32)
    with torch.no_grad():
        for imgs, file_names in data_loader:
            imgs = imgs.to(device)
            image_features = model.encode_image(imgs)
            
            pred_feature = torch.cat((pred_feature, image_features.cpu()))
            nameList += [singleBatch for singleBatch in file_names]
        
        text_inputs = text_inputs.to(device)
        text_features = model.encode_text(text_inputs).cpu().type(torch.float32)  

    pred_feature /= pred_feature.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * pred_feature @ text_features.T).softmax(dim=-1)
    pred_label = torch.argmax(similarity, dim = -1)
    
    csv_name = os.path.basename(save_path)
    csv_root = save_path.replace(csv_name, '')
   
    if not os.path.isdir(csv_root):
        os.makedirs(csv_root)   
    
    csv_dic = {
        "filename" : nameList,
        "label" : pred_label
    }
    pred_df = pd.DataFrame(csv_dic)
    pred_df.to_csv(save_path, index=False)

    print('\n#####################################')
    print('\n          Testing complete!!!          \n')
    print('#####################################')