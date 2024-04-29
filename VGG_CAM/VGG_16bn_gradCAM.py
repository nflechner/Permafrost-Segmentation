#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''

'''
#https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
#https://github.com/tony-mtz/CAM/blob/master/network/net.py 

import torch.nn as nn
import torch
from torchvision import models

class VGG(nn.Module):

    def __init__(self, features, num_classes = 1000, init_weights=False): # INITWEIGHTS SHOULD BE TRUE 
        super(VGG, self).__init__()
        self.features = features[:43]
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(4096, num_classes),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.gradients = None

        if init_weights:
            self._initialize_weights()

    # https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.features(x)
        h = x.register_hook(self.activations_hook)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

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


def make_layers(batch_norm=True):
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
    
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16bn():
    model = VGG(make_layers())
    state_dict = models.VGG16_BN_Weights.DEFAULT.get_state_dict(progress=True)
    model.load_state_dict(state_dict)

    #modify the last two convolutions
    model.features[-7] = nn.Conv2d(512,512,3, padding=1)
    model.features[-4] = nn.Conv2d(512,1,3, padding=1)
    model.features[-3] = nn.BatchNorm2d(1,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    model.classifier = nn.Sequential(
        nn.Linear(1 * 7 * 7, 49),
        nn.ReLU6(True),
        nn.Dropout(0.2),
        nn.Linear(49, 49),
        nn.ReLU6(True),
        nn.Dropout(0.2),
        nn.Linear(49, 1),
        nn.Sigmoid()
    )

    return model
