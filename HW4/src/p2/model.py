import torch.nn as nn
import torchvision.models as models
from utils.helper import load_checkpoint


class Resnet_SSL(nn.Module):
    def __init__(self, class_num, hidden_dim = 4096, pretrained_root = None):
        super().__init__()
        resnet = models.resnet50(pretrained = False)
        if pretrained_root is not None:
            load_checkpoint(pretrained_root, resnet)

        
        resnet.fc = self.net = nn.Sequential(
            nn.Linear(resnet.fc.in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, class_num)
        )
        
        self.resnet = resnet


    def forward(self, x):
        out = self.resnet(x)
        return out


if __name__ == '__main__':
    import torch
    model = Resnet_SSL(65, pretrained_root = '/home/albert/DLCV/HW4/hw4_data/pretrain_model_SL.pt')
    imgs = torch.zeros((10, 3, 128, 128))
    out = model(imgs)
    print(out.shape)
    