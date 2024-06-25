# official package
import argparse, json, os
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm.auto import tqdm
from tokenizers import Tokenizer

# my package
from dataset import P2dataset
from model import VL_model
from utils.helper import fix_seed, load_checkpoint, get_device, process_path, get_key_padding_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default = "/home/albert/DLCV/HW3/hw3_data/p2_data/images/val")
    parser.add_argument('--save_path', default = "output_p2/pred.json")
    args = parser.parse_args()

    data_root = process_path(args.data_root)
    save_path = process_path(args.save_path)

    token_root = './caption_tokenizer.json'
    tokenizer = Tokenizer.from_file(token_root)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    fix_seed(320)
    validset = P2dataset(data_root, transform, mode = 'val')
    
    device = get_device()
    model = VL_model().to(device)
    load_checkpoint('save_model/VL.pth', model)
    

    model.eval()
    pred_list = []
    file_name_list = []
    with torch.no_grad():
        # greedy search
        for img, file_name in tqdm(validset):
            file_name_list.append(file_name)
            
            img = img[None, :].to(device)
            # start with <BOS>
            text = torch.Tensor([[2]]).long().to(device) # (1, 1)
            for i in range(54):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(text.size()[-1]).to(device)
                tgt_key_padding_mask = get_key_padding_mask(text).to(device)

                out = model(img, text, tgt_mask, tgt_key_padding_mask)
                out = out[:, :, -1]
                
                y = torch.argmax(out, dim=1)
                
                text = torch.concat([text, y.unsqueeze(0)], dim=1)

                if y.item() == 3:
                    break

            caption = tokenizer.decode(text[0].cpu().numpy())
            pred_list.append(caption)

    json_name = os.path.basename(save_path)
    json_root = save_path.replace(json_name, '')
   
    if not os.path.isdir(json_root):
        os.makedirs(json_root)  

    output_dict = dict(zip(file_name_list, pred_list))
    with open(save_path, 'w') as fp:
        json.dump(output_dict, fp, indent = 4)
