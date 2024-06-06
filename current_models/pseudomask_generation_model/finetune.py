############
# Imports #
############

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import wandb
import torch.optim.lr_scheduler as lr_scheduler


def FinetuneLoop(model, train_loader, val_loader, lr, weight_decay, lr_gamma, num_epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    loss_function = nn.MSELoss()

    ############################
    # redefine model to finetune
    ############################

    model.classifier = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True), 
            nn.Conv2d(2, 2, kernel_size=3, padding=1))
    model.to(device)
    ########## DO I WANT TO FREEZE LAYERS???


    #######################
    # model training loop #
    #######################

    activation_threshold = 1.5
    min_val_MSE = 1000000

    print('Finetuning model ...')
    for epoch in range(num_epochs):
        print('EPOCH: ',epoch+1)

        ############
        # training #
        ############

        train_loss = 0

        model.train()
        train_batch_counter = 0
        for batch_idx, (images, _, perc_labels) in enumerate(train_loader):     
            train_batch_counter += 1

            # load images and labels 
            images = Variable(images).to(device)  
            perc_labels = Variable(perc_labels.float()).to(device)  

            # train batch   
            outputs = model(images)
            activation_mask = torch.where(outputs> activation_threshold, 1.0, 0.0)
            activated = torch.sum(activation_mask[:,1,:,:], dim=(-1,-2)).float()
            percent_activated = (activated / (activation_mask.shape[-1]* activation_mask.shape[-2])).float()
            percent_activated.requires_grad = True

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
            for batch_idx, (images, _, perc_labels) in enumerate(val_loader):  

                # load images and labels 
                images = Variable(images).to(device)  
                perc_labels = Variable(perc_labels.float()).to(device)  
                outputs = model(images) 
                activation_mask = torch.where(outputs> activation_threshold, 1.0, 0.0)
                activated = torch.sum(activation_mask[:,1,:,:], dim=(-1,-2)).float()
                percent_activated = (activated / (activation_mask.shape[-1]* activation_mask.shape[-2])).float()
                loss = loss_function(percent_activated, perc_labels)

                # update metrics
                val_loss = loss.item()
                running_val_MSE.append(val_loss)

        # lr scheduler step 
        scheduler.step()

        # update metrics
        wandb.log({"train_loss": train_loss / train_batch_counter})
        wandb.log({"val_loss": np.mean(running_val_MSE)})

        # update current best model
        if np.mean(running_val_MSE) < min_val_MSE:
            best_model = model.state_dict()
            min_val_MSE = np.mean(running_val_MSE)

    return best_model

