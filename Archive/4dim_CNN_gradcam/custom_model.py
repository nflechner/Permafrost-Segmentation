
import torch.nn as nn
import torch
from torchvision import models

class model_4D(nn.Module):

    def __init__(self, init_weights=False): # INITWEIGHTS SHOULD BE TRUE 
        super(model_4D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2)
                                )
        self.classifier = nn.Sequential(
            nn.Linear(1 * 12 * 12, 144),
            nn.ReLU6(True),
            nn.Dropout(0.2),
            nn.Linear(144, 72),
            nn.ReLU6(True),
            nn.Dropout(0.2),
            nn.Linear(72, 1),
            nn.Sigmoid()
                        )
        if init_weights:
            self._initialize_weights()

        self.gradients = None

    # https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

    def forward(self, x):
        x = self.features(x)
        h = x.register_hook(self.activations_hook)
        #don't flatten 
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
