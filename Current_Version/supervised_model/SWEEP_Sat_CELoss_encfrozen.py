import torch
from torch import nn
from torchmetrics.functional import jaccard_index
from torchmetrics.functional.classification import multiclass_accuracy
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from transformers import SegformerImageProcessor
import pandas as pd 
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import wandb

# adapted from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Fine_tune_SegFormer_on_custom_dataset.ipynb
class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegformerImageProcessor): image processor to prepare images + segmentation maps.
        """
        self.root_dir = root_dir
        self.image_processor = SegformerImageProcessor(
            image_mean = [74.90, 85.26, 80.06], # use mean calculated over our dataset
            image_std = [15.05, 13.88, 12.01], # use std calculated over our dataset
            do_reduce_labels=False
            )

        self.img_dir = os.path.join(self.root_dir, "images")
        self.ann_dir = os.path.join(self.root_dir, "masks")
        
        # Get all image filenames without extension
        dataframe = pd.read_csv(
            f"{root_dir}/orig_palsa_labels.csv", 
            names=['filename', 'palsa'], 
            header=0
            )
        
        dataframe = dataframe.loc[dataframe['palsa']>0]
        dataframe = dataframe[~dataframe['filename'].str.endswith('aug')]
        checked_names = list(dataframe['filename'])
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f[:-4] in checked_names]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        ann_path = os.path.join(self.ann_dir, f"{img_name}.png")

        image = Image.open(img_path)
        segmentation_map = Image.open(ann_path)

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs

##############
# Custom Loss
##############

def weighted_cross_entropy_loss(logits, targets, class_weights=[1, 6]): # shuld be 1,24
    """
    Calculate weighted cross-entropy loss for binary segmentation using PyTorch's built-in functions.
    
    Args:
    logits (torch.Tensor): Predicted logits with shape [batch, num_classes, height, width]
    targets (torch.Tensor): Ground truth labels with shape [batch, height, width]
    class_weights (list): Weights for each class [weight_class_0, weight_class_1]
    
    Returns:
    torch.Tensor: Weighted cross-entropy loss
    """
    # Ensure inputs are on the same device
    device = logits.device
    targets = targets.to(device)
    
    # Convert class weights to a tensor and move to the same device
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    # Create the loss function with weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    
    # Calculate and return the loss
    return criterion(logits, targets)

# Example usage:
# logits = torch.randn(32, 2, 512, 512)  # [batch, num_classes, height, width]
# targets = torch.randint(0, 2, (32, 512, 512))  # [batch, height, width]
# loss = weighted_cross_entropy_loss(logits, targets)

#########
# CONFIGS
#########

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 300
batch_size = 5
weight_decay = 0.03

# model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
model_name = "sawthiha/segformer-b0-finetuned-deprem-satellite"

# Define the sweep configuration
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'target_jaccard', 'goal': 'maximize'},
    'parameters': {
        'palsa_weight': {'values': [1,8,18]},
        'lr': {'values': [5e-6, 1e-6, 1e-7]}
        }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="Finetune_segformer_sweep")

# Create the full dataset
# root_dir = "/root/Permafrost-Segmentation/Supervised_dataset"
root_dir = "/home/nadjaflechner/Permafrost-Segmentation/Supervised_dataset"
full_dataset = SemanticSegmentationDataset(root_dir)

# Split the dataset into 85% train and 15% validation
total_size = len(full_dataset)
train_size = int(0.85 * total_size)
valid_size = total_size - train_size

train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

def train():

    # Initialize a new wandb run
    run = wandb.init(
        config={
            "model": model_name,   
            "epochs": epochs, 
            "batch_size": batch_size,    
            "weight_decay": weight_decay
            },
        tags=["custom_loss", "frozen_encoder"]
    )

    palsa_weight = wandb.config.palsa_weight
    lr = wandb.config.lr

    # define model
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    ) 

    # Set learnable layers
    for param in model.parameters():
        param.requires_grad = True

    # Freeze encoder layers
    for param in model.segformer.encoder.parameters():
        param.requires_grad = False

    # model to device
    model.to(device)

    # define optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay = weight_decay)

    # Move optimizer to GPU (possibly unneccessary)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    best_jaccard = 0
    best_model_dict = None
    for epoch in range(epochs):
        model.train()
        print(f"Epoch: {epoch}")
        train_loss = []
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:  
            # get the inputs;
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            upsampled_logits = F.interpolate(
                logits.unsqueeze(1).float(), 
                size=[logits.shape[1],labels.shape[-2],labels.shape[-1]], 
                mode="nearest")
            loss = weighted_cross_entropy_loss(upsampled_logits.squeeze(1), labels, class_weights=[1,palsa_weight])
            train_loss.append(loss.detach().cpu())

            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = sum(train_loss) / len(train_loss)
        wandb.log({"train_loss": avg_train_loss})

        model.eval()
        bg_jaccard_scores = []
        target_jaccard_scores = []
        val_loss = []
        overall_accuracy = []

        with torch.no_grad():
            for batch in valid_dataloader:
                # get the inputs;
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                # forward pass
                outputs = model(pixel_values=pixel_values, labels=labels)
                logits = outputs.logits
                upsampled_logits = F.interpolate(
                    logits.unsqueeze(1).float(), 
                    size=[logits.shape[1],labels.shape[-2],labels.shape[-1]], 
                    mode="nearest")
                loss = weighted_cross_entropy_loss(upsampled_logits.squeeze(1), labels, class_weights=[1,palsa_weight])
                val_loss.append(loss.detach().cpu())

                # Convert logits to binary segmentation mask
                predicted = torch.argmax(logits, dim=1)  # Shape: (batch_size, 128, 128)
                
                # Upsample the predicted mask to match the label size
                upsampled_predicted = F.interpolate(
                    predicted.unsqueeze(1).float(), 
                    size=labels.shape[-2:], 
                    mode="nearest"
                )

                # Calculate Jaccard score (IoU) for both classes
                jaccard = jaccard_index(
                    upsampled_predicted.squeeze(1).long(), 
                    labels, 
                    task="multiclass", 
                    num_classes=2, 
                    average='none'
                )
                bg_jaccard_scores.append(jaccard[0])
                target_jaccard_scores.append(jaccard[1])

                # Overall accuracy
                accuracy = multiclass_accuracy(
                    upsampled_predicted.squeeze(1).long(), 
                    labels, 
                    num_classes=2, 
                    average='micro'
                )
                overall_accuracy.append(accuracy)

            avg_val_loss = sum(val_loss) / len(val_loss)
            wandb.log({"val_loss": avg_val_loss})

        avg_bg_jaccard = sum(bg_jaccard_scores) / len(bg_jaccard_scores)
        avg_target_jaccard = sum(target_jaccard_scores) / len(target_jaccard_scores)
        avg_overall_accuracy = sum(overall_accuracy) / len(overall_accuracy)
        wandb.log({"background_jaccard": avg_bg_jaccard})
        wandb.log({"target_jaccard": avg_target_jaccard})
        wandb.log({"avg_overall_accuracy": avg_overall_accuracy})
        print(f"Epoch {epoch}, Average Background Jaccard Score: {avg_bg_jaccard:.4f}, Target Class Jaccard Score: {avg_target_jaccard:.4f}")
        
        # Early stopping check based on target Jaccard score
        if avg_target_jaccard > best_jaccard:
            best_jaccard = avg_target_jaccard
            # Save the best model
            best_model_dict = model.state_dict()


    torch.save(best_model_dict, 'best_model.pth')
    artifact = wandb.Artifact('finetuned_segformer', type='model')
    artifact.add_file('best_model.pth')
    run.log_artifact(artifact)

    torch.cuda.empty_cache()
    del model

wandb.agent(sweep_id, function = train, count = 100)

