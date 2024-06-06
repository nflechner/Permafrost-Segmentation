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
from utils import ImageDataset, TestSet, filter_dataset, imshow_transform
from cnn_classifier import model_4D
from torch.autograd import Variable
from skimage.transform import resize
from skimage.io import imshow
import wandb
import matplotlib.pyplot as plt 
import torch.optim.lr_scheduler as lr_scheduler
import torchmetrics
import json
from pseudomask import Pseudomasks
from train_classifier import ClassifierTrainLoop
from finetune_classifier import FinetuneLoop

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
testset_dir = config_paths.get('testset') 
parent_dir = config_paths.get('data') 
rgb_dir = os.path.join(parent_dir, 'rgb')
hs_dir = os.path.join(parent_dir, 'hs')
dem_dir = os.path.join(parent_dir, 'dem')
labels_file = os.path.join(parent_dir, 'palsa_labels.csv')

# load hyperparams configs dictionary
config_hyperparams = configs.get('hyperparams', {}) 

# assign hyperparams
n_samples = config_hyperparams.get('n_samples')
batch_size = config_hyperparams.get('batch_size')
num_epochs = config_hyperparams.get('num_epochs')
lr = config_hyperparams.get('lr')
lr_gamma = config_hyperparams.get('lr_gamma')
weight_decay = config_hyperparams.get('weight_decay')
finetune = config_hyperparams.get('finetune')

# load data configs dictionary
config_data = configs.get('data', {}) 

# assign data configs
im_size = config_data.get('im_size')
min_palsa_positive_samples = config_data.get('min_palsa_positive_samples')
low_pals_in_val = config_data.get('low_pals_in_val')
augment = config_data.get('augment')
normalize = config_data.get('normalize')
depth_layer = config_data.get('depth_layer')

##########################
# log hyperparams to w&b #
##########################

wandb_tags = [str(tag) for tag in [low_pals_in_val, augment, normalize, depth_layer] if tag]

run = wandb.init(
    # Set the project where this run will be logged
    project="VGG_CAMs",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "lr_gamma": lr_gamma,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "n_samples": n_samples,
        "finetune": finetune,
        "weight_decay": weight_decay,
        "im_size": im_size,
        "min_palsa_positive_samples": min_palsa_positive_samples,
        "augment": augment,
        "normalize": normalize,
        "low_pals_in_val": low_pals_in_val,
        "depth_layer": depth_layer
            },
    tags= wandb_tags
)

#########################
# configure dataloaders #
#########################

train_files, val_files = filter_dataset(labels_file, augment, min_palsa_positive_samples, low_pals_in_val, n_samples)
# choose depth data based on configs
depth_dir = hs_dir if depth_layer == "hs" else dem_dir
# Create the datasets and data loaders
train_dataset = ImageDataset(depth_dir, rgb_dir, train_files, im_size, normalize)
val_dataset = ImageDataset(depth_dir, rgb_dir, val_files, im_size, normalize)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


################
# define model #
################

model = model_4D()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


##############################
# Train model on binary data #
##############################

best_model = ClassifierTrainLoop(model, train_loader, val_loader, 
                                 lr, weight_decay, lr_gamma, num_epochs)
# after all epochs, save the best model as an artifact to wandb
torch.save(best_model, '/home/nadjaflechner/models/model.pth')
artifact = wandb.Artifact('classification_model', type='model')
artifact.add_file('/home/nadjaflechner/models/model.pth')
run.log_artifact(artifact)


############################
# Finetune model on % data #
############################

if finetune:
    # use trained model from above
    model.load_state_dict(best_model)
    # finetune pretrained model
    finetuned_model = FinetuneLoop(model, train_loader, val_loader, 
                                 lr, weight_decay, lr_gamma, num_epochs)
    # after all epochs, save the best model as an artifact to wandb
    torch.save(finetuned_model, '/home/nadjaflechner/models/model.pth')
    artifact = wandb.Artifact('finetuned_model', type='model')
    artifact.add_file('/home/nadjaflechner/models/model.pth')
    run.log_artifact(artifact)


##################
# evaluate model #
##################

test_set = TestSet(depth_layer, testset_dir, normalize)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)

pseudomask_generator = Pseudomasks()
pseudomask_generator.model_from_dict(best_model)

for i in range(len(test_loader.dataset)):
    im, lab, perc_label, gt_mask = next(iter(test_loader))

    # currently not yet comparing negative samples 
    if not lab == 0:
        pseudomask = pseudomask_generator.generate_mask(
            im, gt_mask, 
            save_plot = True,
            cam_threshold_factor = 0.5, 
            overlap_threshold= 0.5,
            snic_seeds = 100,
            snic_compactness = 10)
        
        # calculate metrics to evaluate model on test set
        generated_mask = torch.Tensor(pseudomask).int().view(400,400)
        groundtruth_mask = torch.Tensor(gt_mask).int().view(400,400)
        metrics = pseudomask_generator.calc_metrics(generated_mask, groundtruth_mask)


##############
# finish run #
##############

wandb.finish()