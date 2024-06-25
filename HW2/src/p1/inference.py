# official package
import os, argparse 
import torch

# my package
from utils.helper import get_device, generate_output,load_checkpoint
from model import Generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default = "./output_p1")
    args = parser.parse_args()
    save_path = os.path.expanduser(args.save_path)
    
    if not os.path.isabs(save_path):
        save_path = os.path.join('..', save_path)

    device = get_device()

    num = 1000
    noise_dim = 100
    noise = torch.load('./noise_dc.pth')
   
    G = Generator(noise_dim = noise_dim).to(device)
    load_checkpoint('./save_model/DCGAN_G.pth', G)

    print('#####################################')
    print('\n          Testing start!!!          \n')
    print('#####################################')
    generate_output(noise_dim, num, G, device, save_path, noise)
    print('\n#####################################')
    print('\n          Testing complete!!!          \n')
    print('#####################################')