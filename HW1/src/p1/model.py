import torch.nn as nn
import torchvision.models as models
from torch.nn.init import kaiming_normal_


class Resnext101(nn.Module):
    def __init__(self, num_out):
        super().__init__()
        self.layer = nn.Sequential(
            models.resnext101_32x8d(pretrained = False),
            nn.ReLU(),
            nn.Linear(1000, num_out)
        )
        
    def forward(self, x):
        x = self.layer(x)
        return x
    
    
class VGG16(nn.Module):
    def __init__(self, num_out):
        super(VGG16, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
    
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 1, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 1, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 1, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512*8*8, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(4096, num_out)
        )
        
        for m in self.modules():	
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data, nonlinearity='relu')
                
        
    def forward(self, x):
        x = self.cnn_layers(x) 
        x = x.view(-1, 512*8*8)
        out = self.fc_layers(x)  
        
        return out 
    
    
if __name__ == '__main__': 
    import torch 
    
    x = torch.zeros((10, 3, 224, 224))
    model = VGG16(50)
    out = model(x)
    print(out.shape)
   