# official package
import os, argparse 
import torch
from torchvision import transforms
import torch.optim as optim

# my package
from utils.helper import get_device, fix_seed, load_checkpoint, get_schedule, generate_imgs
from model import UNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default = "./output_p2")
    args = parser.parse_args()
    save_path = os.path.expanduser(args.save_path)
    
    if not os.path.isabs(save_path):
        save_path = os.path.join('..', save_path)

    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img * 0.5 + 0.5),
        transforms.ToPILImage(),
        transforms.Resize((28,28)),
    ])

    fix_seed(320)
    device = get_device()

    step = 400
    model = UNet(T = step, num_labels=10, ch=64, ch_mult=[1, 2, 2, 2],
                 num_res_blocks=2, dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 2e-4, betas = (0.9, 0.999))
    load_checkpoint('./save_model/ddpm.pth', model, optimizer)
    
    print('#####################################')
    print('\n          Testing start!!!          \n')
    print('#####################################')
    num_per_digit = 100
    num_class = 10
    params = get_schedule(T = step)
    with torch.no_grad():
        imgs = generate_imgs(model, step, device, params)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for digits in range(num_class):
            for i in range(num_per_digit):
                img = imgs[digits*num_per_digit + i]
                
                img = reverse_transforms(img)
                image_path = os.path.join(save_path, str(digits) + "_" + str(i+1).zfill(3) + ".png")
                img.save(image_path)
    print('\n#####################################')
    print('\n          Testing complete!!!          \n')
    print('#####################################')
