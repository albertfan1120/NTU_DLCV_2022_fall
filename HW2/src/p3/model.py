import torch
import torch.nn as nn
from torch.autograd import Function


class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True),
        )
        
        self.feature_extractor_linear = nn.Sequential(
            nn.Linear(800, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.ReLU(True)
        )
        
        self.label_classifier = nn.Sequential(
            nn.Linear(512, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),

            nn.Linear(100, 10),
        )

        self.domain_classifier = nn.Sequential(       
            nn.Linear(512, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),

            nn.Linear(100, 2), 
        )

        self.feature_extractor.apply(weights_init)
        self.label_classifier.apply(weights_init)
        self.domain_classifier.apply(weights_init)


    def forward(self, x, alpha = 0.42, reverse = True):       
        features = self.feature_extractor(x)
        features = features.view(features.shape[0], -1)
        features =  self.feature_extractor_linear(features)
        # predict class
        label = self.label_classifier(features)

        # predict domain
        if reverse: 
            reversed_feature = ReverseLayerF.apply(features, alpha)
            domain = self.domain_classifier(reversed_feature)
        else:
            domain = self.domain_classifier(features)


        return label, domain
    

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        
        return output, None
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')


if __name__ == '__main__':
    model = DANN()
    x = torch.zeros((10, 3, 28, 28))
    label, domain = model(x) 
    print(label.shape, domain.shape)
    