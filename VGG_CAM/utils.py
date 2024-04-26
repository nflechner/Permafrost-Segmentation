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

