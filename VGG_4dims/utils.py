import torch
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import rasterio
import numpy as np
from torchvision import models, transforms
import matplotlib.pyplot as plt
from torch import Tensor

class ImageDataset(Dataset):
    def __init__(self, hs_dir, RGB_dir, labels_df):
        self.RGB_dir = RGB_dir
        self.hs_dir = hs_dir
        self.labels_df = labels_df

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.index[idx]
        RGB_img_path = os.path.join(self.RGB_dir, f"{img_name}.tif")
        hs_img_path = os.path.join(self.hs_dir, f"{img_name}_hs.tif")

        with rasterio.open(RGB_img_path) as RGB_src:
            # Read the image data
            RGB_img = RGB_src.read()

        with rasterio.open(hs_img_path) as hs_src:
            # Read the image data
            hs_img = hs_src.read()

        # convert and upsample hs image
        hs_image_array = np.array(hs_img)
        hs_image_tensor = torch.from_numpy(hs_image_array)
        hs_image_tensor = hs_image_tensor.float()
        bilinear = nn.Upsample(size=200, mode='bilinear')
        hs_upsampled_tensor = bilinear(hs_image_tensor.unsqueeze(0)).squeeze(0) 

        # converting RGB to tensor
        RGB_image_array = np.array(RGB_img)
        RGB_image_tensor = torch.from_numpy(RGB_image_array)
        RGB_image_tensor = RGB_image_tensor.float()

        combined_tensor = torch.concatenate((RGB_image_tensor, hs_upsampled_tensor))

        label = self.labels_df.iloc[idx, 0]
        label = 1 if label > 0 else 0

        return combined_tensor, label

#https://www.fast.ai/
#fastai code snippet
class SaveFeatures():
    features=None
    def __init__(self,m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

# https://github.com/tony-mtz/CAM/blob/master/network/utils.py
# def accuracy(input:Tensor, targs:Tensor):
#     n = targs.shape[0]
#     input = input.argmax(dim=-1).view(n,-1)
#     targs = targs.view(n,-1)
#     return (input==targs).float().mean().cpu().detach().numpy()

def accuracy(outputs:Tensor, labels:Tensor):
    conv_outputs = torch.where(outputs.squeeze() > 0.5, 1.0, 0.0)
    return (conv_outputs==labels).float().mean().detach().cpu().numpy()


#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def imshow_transform(image_in, title=None):
    """Imshow for Tensor."""
    img = np.rollaxis(image_in.squeeze().cpu().detach().long().numpy(),0,3)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    return img