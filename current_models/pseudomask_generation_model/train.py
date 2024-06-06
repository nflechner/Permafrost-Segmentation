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
import torchmetrics

def ClassifierTrainLoop(model, train_loader, val_loader, lr, weight_decay, lr_gamma, num_epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    loss_function = nn.CrossEntropyLoss()

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

    return best_model

