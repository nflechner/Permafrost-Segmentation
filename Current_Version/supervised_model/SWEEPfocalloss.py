import torch
from torch import nn
from torchmetrics.functional import jaccard_index
from torchmetrics.functional.classification import multiclass_accuracy
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
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
            do_reduce_labels=True
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


#########
# CONFIGS
#########

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 400
warmup_steps = 100

# Early stopping parameters
patience = 10

#########
# SWEEP 1
#########

# model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
model_name = "sawthiha/segformer-b0-finetuned-deprem-satellite"

# Define the sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {'name': 'target_jaccard', 'goal': 'maximize'},
    'parameters': {
        'freeze_encoder': {'values': [True, False]},
        'lr': {"max": 1e-5, "min": 5e-8},
        'FL_alpha': {"max": 0.8, "min": 0.2},
        'FL_gamma': {"max": 4, "min": 1},
        'weight_decay': {"max": 0.07, "min": 0.01},
        'batch_size': {"max": 16, "min": 4}
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

def train():

    # Initialize a new wandb run
    run = wandb.init(
        config={
            "model": model_name,   
            "epochs": epochs 
            },
        tags=["sigmoid_focal_loss"]
    )

    freeze_encoder = wandb.config.freeze_encoder
    lr = wandb.config.lr
    FL_alpha = wandb.config.FL_alpha
    FL_gamma = wandb.config.FL_gamma
    weight_decay = wandb.config.weight_decay
    batch_size = wandb.config.batch_size

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    # define model
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=1,
        ignore_mismatched_sizes=True
    ) 

    # Set learnable layers
    for param in model.parameters():
        param.requires_grad = True

    # Freeze encoder layers
    if freeze_encoder == True:
        for param in model.segformer.encoder.parameters():
            param.requires_grad = False

    # # Optionally, unfreeze the last few layers of the encoder
    # # Adjust the number of unfrozen blocks as needed
    # num_unfrozen_blocks = 4
    # for i in range(len(model.segformer.encoder.block) - num_unfrozen_blocks, len(model.segformer.encoder.block)):
    #     for param in model.segformer.encoder.block[i].parameters():
    #         param.requires_grad = True

    # model to device
    model.to(device)

    # define optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay = weight_decay)

    # define scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Move optimizer to GPU (possibly unneccessary)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    best_jaccard = 0
    epochs_no_improve = 0
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
            labels = torch.where(labels == 255, 0, 1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            upsampled_logits = F.interpolate(
                logits.unsqueeze(1).float(), 
                # size=[logits.shape[1],labels.shape[-2],labels.shape[-1]], 
                size=[1, labels.shape[-2],labels.shape[-1]], 
                mode="nearest")
            loss = sigmoid_focal_loss(
                upsampled_logits.squeeze(1), 
                labels.unsqueeze(1).float(), 
                alpha = FL_alpha, 
                gamma = FL_gamma, 
                reduction="mean"
                )
            train_loss.append(loss.detach().cpu())

            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

            # Update progress bar
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", 
                 "LR": f"{scheduler.get_last_lr()[0]:.6f}"}
                 )

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
                labels = torch.where(labels == 255, 0, 1)

                # forward pass
                outputs = model(pixel_values=pixel_values, labels=labels)
                logits = outputs.logits

                upsampled_logits = F.interpolate(
                    logits.unsqueeze(1).float(), 
                    # size=[logits.shape[1],labels.shape[-2],labels.shape[-1]], 
                    size=[1, labels.shape[-2],labels.shape[-1]], 
                    mode="nearest")
                loss = sigmoid_focal_loss(
                    upsampled_logits.squeeze(1), 
                    labels.unsqueeze(1).float(), 
                    alpha = FL_alpha,
                    gamma = FL_gamma,
                    reduction="mean"
                    )
                val_loss.append(loss.detach().cpu())

                # Convert logits to binary segmentation mask
                predicted = torch.sigmoid(logits)  # Shape: (batch_size, 128, 128)
                predicted = torch.where(predicted > 0.5, 1, 0)  # Shape: (batch_size, 128, 128)
                
                # Upsample the predicted mask to match the label size
                upsampled_predicted = F.interpolate(
                    predicted.unsqueeze(1).float(), 
                    size=[1, labels.shape[-2],labels.shape[-1]], 
                    mode="nearest"
                )

                # Calculate Jaccard score (IoU) for both classes
                jaccard = jaccard_index(
                    upsampled_predicted.squeeze(1,2).long(), 
                    labels, 
                    task="multiclass", 
                    num_classes=2, 
                    average='none'
                )
                bg_jaccard_scores.append(jaccard[0])
                target_jaccard_scores.append(jaccard[1])

                # Overall accuracy
                accuracy = multiclass_accuracy(
                    upsampled_predicted.squeeze(1,2).long(), 
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
            epochs_no_improve = 0
            # Save the best model
            best_model_dict = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered. No improvement in target Jaccard score for {patience} epochs.")
                break

    torch.save(best_model_dict, 'best_model.pth')
    artifact = wandb.Artifact('finetuned_segformer', type='model')
    artifact.add_file('best_model.pth')
    run.log_artifact(artifact)

    torch.cuda.empty_cache()
    del model

wandb.agent(sweep_id, function = train, count = 200)

