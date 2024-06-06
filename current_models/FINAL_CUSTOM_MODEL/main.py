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
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
loss_function = nn.CrossEntropyLoss()

#######################
# model training loop #
#######################

# define metrics
Accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)
F1 = torchmetrics.F1Score(task="multiclass", num_classes=2).to(device)
Recall = torchmetrics.Recall(task="multiclass", average="micro", num_classes=2).to(device)

max_val_F1 = 0

for epoch in range(num_epochs):
    print('EPOCH: ',epoch+1)

    ############
    # training #
    ############

    train_loss = 0
    train_accuracy = 0 
    train_Recall = 0 
    train_F1 = 0 

    model.train()
    train_batch_counter = 0
    for batch_idx, (images, labels, perc_labels) in enumerate(train_loader):     
        train_batch_counter += 1

        # load images and labels 
        images = Variable(images).to(device)  
        labels = Variable(labels.long()).to(device)  

        # train batch   
        outputs = model(images) 
        optimizer.zero_grad()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()  

        # update metrics
        train_loss += loss.item()
        train_accuracy += Accuracy(outputs.softmax(dim=-1), labels)
        train_Recall += Recall(outputs.softmax(dim=-1), labels)
        train_F1 += F1(outputs.softmax(dim=-1), labels)

    ##############
    # validation #
    ##############

    val_loss = 0
    val_accuracy = 0 
    val_Recall = 0 
    val_F1 = 0 

    running_val_F1 = []
    model.eval()
    val_batch_counter = 0
    with torch.no_grad():
        for batch_idx, (images, labels, perc_labels) in enumerate(val_loader):  
            val_batch_counter += 1

            # load images and labels 
            images = Variable(images).to(device)  
            labels = Variable(labels.long()).to(device)  
            outputs = model(images) 
            loss = loss_function(outputs, labels)

            # update metrics
            val_loss += loss.item()
            val_accuracy += Accuracy(outputs.softmax(dim=-1), labels)
            val_Recall += Recall(outputs.softmax(dim=-1), labels)
            val_F1 += F1(outputs.softmax(dim=-1), labels)

            # handle F1 separately for best model selection
            f1 = F1(outputs.softmax(dim=-1), labels)
            running_val_F1.append(f1.detach().cpu().numpy())
            val_F1 = f1

    # lr scheduler step 
    scheduler.step()

    # update metrics
    wandb.log({"val_loss": val_loss / val_batch_counter})
    wandb.log({"val_accuracy": val_accuracy / val_batch_counter})
    wandb.log({"val_Recall": val_Recall / val_batch_counter})
    wandb.log({"val_F1": val_F1 / val_batch_counter})

    wandb.log({"train_loss": train_loss / train_batch_counter})
    wandb.log({"train_accuracy": train_accuracy / train_batch_counter})
    wandb.log({"train_Recall": train_Recall / train_batch_counter})
    wandb.log({"train_F1": train_F1 / train_batch_counter})

    # update current best model
    if np.mean(running_val_F1) > max_val_F1:
        best_model = model.state_dict()
        max_val_F1 = np.mean(running_val_F1)

# after all epochs, save the best model as an artifact to wandb
torch.save(best_model, '/home/nadjaflechner/models/model.pth')
artifact = wandb.Artifact('classification_model', type='model')
artifact.add_file('/home/nadjaflechner/models/model.pth')
run.log_artifact(artifact)


############################
# Finetune model on % data #
############################

if finetune:

    ################
    # define model #
    ################

    model.load_state_dict(best_model)
    model.to(device)
    model.classifier = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True), 
            nn.Conv2d(2, 2, kernel_size=3, padding=1)
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    loss_function = nn.MSELoss()

    #######################
    # model training loop #
    #######################

    activation_threshold = 1.5
    min_val_MSE = 1000000

    for epoch in range(num_epochs):
        print('EPOCH: ',epoch+1)

        ############
        # training #
        ############

        train_loss = 0

        model.train()
        train_batch_counter = 0
        for batch_idx, (images, labels, perc_labels) in enumerate(train_loader):     
            train_batch_counter += 1

            # load images and labels 
            images = Variable(images).to(device)  
            labels = Variable(labels.long()).to(device)  

            # train batch   
            outputs = model(images)
            activation_mask = torch.where(outputs> activation_threshold, 1.0, 0.0)
            activated = torch.sum(activation_mask[:,1,:,:], dim=(-1,-2))
            percent_activated = activated / (activation_mask.shape[-1]* activation_mask.shape[-2])

            optimizer.zero_grad()
            loss = loss_function(percent_activated, perc_labels)
            loss.backward()
            optimizer.step()  

            # update metrics
            train_loss += loss.item()

        ##############
        # validation #
        ##############

        running_val_MSE = []
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels, perc_labels) in enumerate(val_loader):  

                # load images and labels 
                images = Variable(images).to(device)  
                labels = Variable(labels.long()).to(device)  
                outputs = model(images) 
                outputs = model(images)
                activation_mask = torch.where(outputs> activation_threshold, 1.0, 0.0)
                activated = torch.sum(activation_mask[:,1,:,:], dim=(-1,-2))
                percent_activated = activated / (activation_mask.shape[-1]* activation_mask.shape[-2])
                loss = loss_function(percent_activated, perc_labels)

                # update metrics
                val_loss = loss.item()
                running_val_MSE.append(val_loss.detach().cpu().numpy())

        # lr scheduler step 
        scheduler.step()

        # update metrics
        wandb.log({"train_loss": train_loss / train_batch_counter})
        wandb.log({"val_loss": np.mean(running_val_MSE)})

        # update current best model
        if np.mean(running_val_MSE) < min_val_MSE:
            best_model = model.state_dict()
            max_val_F1 = np.mean(running_val_MSE)

    # after all epochs, save the best model as an artifact to wandb
    torch.save(best_model, '/home/nadjaflechner/models/model.pth')
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