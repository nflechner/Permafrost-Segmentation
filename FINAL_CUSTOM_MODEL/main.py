############
# Imports #
############

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
from utils import ImageDataset, SaveFeatures, accuracy, imshow_transform
from custom_model import model_4D
from torch.autograd import Variable
from skimage.transform import resize
from skimage.io import imshow
import wandb
import matplotlib.pyplot as plt 
import torch.optim.lr_scheduler as lr_scheduler
import json

##################
## load configs ##
##################

config_path = os.path.join(os.getcwd(), 'configs.json')
with open(config_path, 'r') as config_file:
    configs = json.load(config_file)

# load paths configs dictionary
config_paths = configs.get('paths', {}) 

# assign paths
palsa_shapefile = config_paths.get('palsa_shapefile') 
parent_dir = config_paths.get('data') 
rgb_dir = os.path.join(parent_dir, 'rgb')
hs_dir = os.path.join(parent_dir, 'hs')
dem_dir = os.path.join(parent_dir, 'dem')
labels_file = os.path.join(parent_dir, 'palsa_labels.csv')

# load hyperparams configs dictionary
config_hyperparams = configs.get('hyperparams', {}) 

# assign hyperparams
n_samples = config_hyperparams.get('n_samples')
n_samples_train = int(round(n_samples*0.8))
n_samples_val = int(round(n_samples*0.2))
batch_size = config_hyperparams.get('batch_size')
num_epochs = config_hyperparams.get('num_epochs')
lr = config_hyperparams.get('lr')
weight_decay = config_hyperparams.get('weight_decay')

# load data configs dictionary
config_data = configs.get('data', {}) 

# assign data configs
im_size = config_data.get('im_size')
min_palsa_positive_samples = config_data.get('min_palsa_positive_samples')
low_pals_in_val = config_data.get('low_pals_in_val')
augment = config_data.get('augment')
normalize = config_data.get('normalize')


##########################
# log hyperparams to w&b #
##########################

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
        "im_size": im_size,
        "min_palsa_positive_samples": min_palsa_positive_samples
    },
    tags=['4D', 'LRScheduler', 'MSELoss', 'TrainFromScratch']
)