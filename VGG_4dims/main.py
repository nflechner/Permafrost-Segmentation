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
from utils import ImageDataset
from VGG_model import vgg19


#########################################################################

RGB_dir = None # TO BE FILLED
hs_dir = None # TO BE FILLED
labels_file = None # TO BE FILLED

# Load the labels from the CSV file
labels_df = pd.read_csv(labels_file, index_col=0).head(100)

# Split the dataset into training and validation sets
train_df = labels_df.head(80)
val_df = labels_df.drop(train_df.index)

# Create the datasets and data loaders
train_dataset = ImageDataset(hs_dir, RGB_dir, train_df)
val_dataset = ImageDataset(hs_dir, RGB_dir, val_df)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


#########################################################################


model = vgg19()

#freeze layers
for param in model.parameters():
    param.requires_grad = False

#modify the last two convolutions
model.features[-5] = nn.Conv2d(512,512,3, padding=1)
model.features[-3] = nn.Conv2d(512,2,3, padding=1)

#remove fully connected layer and replace it with AdaptiveAvePooling
model.classifier = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(),
                                nn.LogSoftmax()
                                )

########################################################################

weights = models.VGG19_Weights.DEFAULT
transforms = weights.transforms()


# DON'T FORGET TO TRANSFORM BATCH!




for imgs, labels in train_loader:
    first_batch = imgs
    first_labels = labels
    break

transformed_batch = transforms(first_batch)

prediction = VGG(transformed_batch).softmax(1)
class_scores, class_indices = torch.max(prediction, dim=1)