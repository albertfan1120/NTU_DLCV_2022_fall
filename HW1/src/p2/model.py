import torch
import torch.nn as nn
import torchvision.models as models


class VGG16_FCN32(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        vgg = models.vgg16(pretrained = True)

        self.feature = vgg.features # (512, 16, 16)
        self.fc32 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size = 2),
            nn.ReLU(inplace = True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, num_class, kernel_size = 1),
            nn.ConvTranspose2d(num_class, num_class, 64, 32, 0, bias=False),
        )
        
    def forward(self, x):
        x = self.feature(x) 
        x = self.fc32(x)
        return x
  
################## PSPnet #######################
class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)


    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(nn.Upsample(x_size[2:], mode = 'bilinear', align_corners = True)(f(x)))
        out = torch.cat(out, 1)
        return out


class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        resnet = models.resnet101(pretrained=False)
       
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        
        initialize_weights(self.ppm, self.final)


    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        
        return nn.Upsample(x_size[2:], mode = 'bilinear', align_corners = True)(x)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


if __name__ == '__main__':
    x = torch.zeros((10, 3, 512, 512))
    model = PSPNet(7)
    out = model(x)
    print(out.shape)