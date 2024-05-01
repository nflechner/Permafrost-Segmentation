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
    def __init__(self, image_dir, labels_df):
        self.image_dir = image_dir
        self.labels_df = labels_df

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.index[idx]
        img_path = os.path.join(self.image_dir, f"{img_name}.tif")

        # Open the TIF image using rasterio
        with rasterio.open(img_path) as src:
            # Read the image data
            image_data = src.read()
        image_array = np.array(image_data)
        image_tensor = torch.from_numpy(image_array)
        image_tensor = image_tensor.float()

        label = self.labels_df.iloc[idx, 0]

        return image_tensor, label

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