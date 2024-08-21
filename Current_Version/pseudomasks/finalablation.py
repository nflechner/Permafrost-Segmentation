############
# Imports #
############

import numpy as np
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchmetrics
import wandb
from torch.autograd import Variable

import json
import os

from torch.utils.data import DataLoader

from model.cnn_classifier import model_4D
from model.finetune import FinetuneLoop
from model.pseudomask import Pseudomasks
from utils.data_modules import ImageDataset, TestSet, filter_dataset


def ClassifierTrainLoop(model, train_loader, val_loader, lr, weight_decay, lr_gamma, num_epochs,loss_weights):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    loss_function = nn.CrossEntropyLoss(weight = torch.tensor([1.19, 6.19]).to(device))

    # define metrics
    Accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)
    F1 = torchmetrics.F1Score(task="multiclass", num_classes=2).to(device)
    Recall = torchmetrics.Recall(task="multiclass", average="micro", num_classes=2).to(device)

    max_val_F1 = 0

    print('Training model ...')
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
        for images, labels, _,_ in train_loader:
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

        model.eval()
        val_batch_counter = 0
        with torch.no_grad():
            for images, labels, _,_ in val_loader:
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
        if (val_F1 / val_batch_counter) > max_val_F1:
            best_model = model.state_dict()
            max_val_F1 = val_F1 / val_batch_counter

    del images
    del labels

    return best_model

##############################
# hardcoded constant configs # 
##############################

config_path = os.path.join(os.getcwd(), 'configs/classifier_configs.json')
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

# constant params
cam_threshold_factor = 0.85
overlap_threshold = 0.45
snic_seeds = 100
snic_compactness = 4
std_from_mean = 0
finetune = False
batch_size = 20
num_epochs = 20
im_size = 200
low_pals_in_val = False
weight_decay = 0.04
lr_gamma = 0.93

########################
# define run 3 configs #
########################

# assign hyperparams
min_palsa_positive_samples = 10
n_samples = 10205 
loss_weights = [1.16,7.12]

normalize = False
augment = False
depth_layer = 'hs'
lr = 0.00001

run = wandb.init(
    # Track hyperparameters and run metadata
    project="Permafrost_Ablation",
    config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "n_samples": n_samples,
        "finetune": finetune,
        "im_size": im_size,
        "augment": augment,
        "normalize": normalize,
        "low_pals_in_val": low_pals_in_val,
        "depth_layer": depth_layer,
        "min_palsa_positive_samples": min_palsa_positive_samples            
        },
        tags=['PseudomaskGridsearch']
)

# configure dataloaders #
train_files, val_files = filter_dataset(labels_file, augment, min_palsa_positive_samples, low_pals_in_val, n_samples)
# choose depth data based on configs
depth_dir = hs_dir if depth_layer == "hs" else dem_dir
# Create the datasets and data loaders
train_dataset = ImageDataset(depth_dir, rgb_dir, train_files, im_size, normalize)
val_dataset = ImageDataset(depth_dir, rgb_dir, val_files, im_size, normalize)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# define model #
model = model_4D()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train model on binary data #
best_model = ClassifierTrainLoop(model, train_loader, val_loader,
                                lr, weight_decay, lr_gamma, num_epochs, loss_weights)
# after all epochs, save the best model as an artifact to wandb
torch.save(best_model, '/home/nadjaflechner/models/model.pth')
artifact = wandb.Artifact('classification_model', type='model')
artifact.add_file('/home/nadjaflechner/models/model.pth')
run.log_artifact(artifact)

# evaluate model 
print('Testing model ...')
test_set = TestSet(depth_layer, testset_dir, normalize)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)

pseudomask_generator = Pseudomasks(test_loader, cam_threshold_factor, overlap_threshold,
                                    snic_seeds, snic_compactness, finetune, std_from_mean)
pseudomask_generator.model_from_dict(best_model)
pseudomask_generator.test_loop(test_loader)
