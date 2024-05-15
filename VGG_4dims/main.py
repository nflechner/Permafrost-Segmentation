############
# Imports #

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
from utils import ImageDataset, SaveFeatures, accuracy, imshow_transform
from VGG_16bn import vgg16bn
from torch.autograd import Variable
from skimage.transform import resize
from skimage.io import imshow
import wandb
import matplotlib.pyplot as plt 

# %matplotlib inline


###################
# Hyperparameters #

n_samples = 25000
n_samples_train = int(round(n_samples*0.8))
batch_size = 50
current_computer = "ubuntu" # "macbook"
layers_to_freeze = 41
lr = 0.00001
weight_decay=0.09
num_epochs = 15
im_size = 100


##########################
# log hyperparams to w&b #

run = wandb.init(
    # Set the project where this run will be logged
    project="VGG_CAMs",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "n_samples": n_samples,
        "layers_to_freeze": layers_to_freeze,
        "weight_decay": weight_decay,
        "im_size": im_size
    },
)

#############
# Load data #

if current_computer == "ubuntu":
    image_dir = f"/home/nadjaflechner/Palsa_data/dataset_{im_size}m/"
    labels_file = f"/home/nadjaflechner/Palsa_data/binary_palsa_labels_{im_size}m.csv"
elif current_computer == "macbook":
    image_dir = f"/Users/nadja/Documents/UU/Thesis/Data/{im_size}m"
    labels_file = f"/Users/nadja/Documents/UU/Thesis/Data/{im_size}m_palsa_labels.csv"

# Load the labels from the CSV file
labels_df = pd.read_csv(labels_file, index_col=0).head(n_samples)

# Split the dataset into training and validation sets
train_df = labels_df.head(n_samples_train)
val_df = labels_df.drop(train_df.index)

# Create the datasets and data loaders
train_dataset = ImageDataset(image_dir, train_df )
val_dataset = ImageDataset(image_dir, val_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


################
# Define model #save/to/path/model

model = vgg16bn()

#freeze layers
for idx, param in enumerate(model.parameters()):
    if idx == layers_to_freeze:
        break
    param.requires_grad = False

#modify the last two convolutions
model.features[-7] = nn.Conv2d(512,512,3, padding=1)
model.features[-4] = nn.Conv2d(512,2,3, padding=1)
model.features[-3] = nn.BatchNorm2d(2,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

#remove fully connected layer and replace it with AdaptiveAvePooling
model.classifier = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1),
                                nn.Flatten()
                                )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_function = nn.CrossEntropyLoss()

##################
# Training loop #

# adapted from https://github.com/tony-mtz/CAM/blob/master/network/net.py

weights = models.VGG16_BN_Weights.DEFAULT
transforms = weights.transforms()

mean_train_losses = []
mean_val_losses = []

mean_train_acc = []
mean_val_acc = []

max_val_acc = 0

for epoch in range(num_epochs):
    print('EPOCH: ',epoch+1)

    train_acc = []
    val_acc = []

    running_loss = 0.0
    model.train()
    train_batch_count = 0
    for batch_idx, (images, labels) in enumerate(train_loader):     

        # load images and labels 
        images = Variable(transforms(images)).to(device)
        labels = Variable(labels.long()).to(device)

        # train batch   
        outputs = model(images)         
        optimizer.zero_grad()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()  

        # calculate loss and accuracy
        train_acc.append(accuracy(outputs, labels.long()))
        running_loss += loss.item()
        train_batch_count += 1
    
    model.eval()
    val_batch_count = 0
    val_running_loss = 0.0
    model.to(device)

    for batch_idx, (images, labels) in enumerate(val_loader):     
        # inference   
        images = Variable(transforms(images)).to(device)
        labels = Variable(labels.long()).to(device)
        outputs = model(images)
        loss = loss_function(outputs, labels)

        val_acc.append(accuracy(outputs, labels))
        val_running_loss += loss.item()
        val_batch_count +=1

    # update losses and accuracies 

    mean_train_acc.append(np.mean(train_acc))
    mean_val_acc.append(np.mean(val_acc))
    mean_train_losses.append(running_loss/train_batch_count)
    mean_val_losses.append(val_running_loss/val_batch_count)

    wandb.log({"train_accuracy": np.mean(train_acc)})
    wandb.log({"val_accuracy": np.mean(val_acc)})
    wandb.log({"train_loss": running_loss/train_batch_count})
    wandb.log({"val_loss": val_running_loss/val_batch_count})

    if np.mean(val_acc) > max_val_acc:
        best_model = model.state_dict()
        max_val_acc = np.mean(val_acc)

torch.save(best_model, '/home/nadjaflechner/models/model.pth')
artifact = wandb.Artifact('model', type='model')
artifact.add_file('/home/nadjaflechner/models/model.pth')
run.log_artifact(artifact)
