import numpy as np
import os
import pandas as pd 
import torch
import torch.nn.functional as F
import wandb

from PIL import Image
from torch import nn
from torchmetrics.functional import jaccard_index
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.utils.data import Dataset, random_split, DataLoader

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


#########
# CONFIGS
#########

model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
# model_name = "sawthiha/segformer-b0-finetuned-deprem-satellite"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 4
lr = 7e-5
warmup_steps = 100  # Adjust this value as needed

# Early stopping parameters
patience = 5
best_jaccard = 0
epochs_no_improve = 0

run = wandb.init(
    # Set the project where this run will be logged
    project="Finetune_segformer",
    # Track hyperparameters and run metadata
    config={
        "epochs": epochs,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "patience": patience,
        "model": model_name,
        }
)

###################
# Generate Datasets
###################

# Create the full dataset
root_dir = "/root/Permafrost-Segmentation/Supervised_dataset"
full_dataset = SemanticSegmentationDataset(root_dir)

# Split the dataset into 85% train and 15% validation
total_size = len(full_dataset)
train_size = int(0.85 * total_size)
valid_size = total_size - train_size

train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

train_dataloader = DataLoader(train_dataset, batch_size=5)#, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=5)

# define model
model = SegformerForSemanticSegmentation.from_pretrained(
    model_name,
    num_labels=2,
    ignore_mismatched_sizes=True
) 

# Set learnable layers
for param in model.parameters():
    param.requires_grad = True

# model to device
model.to(device)

# define optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay = 0.03)

# define scheduler
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# Move optimizer to GPU (possibly unneccessary)
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)


#########
# TRAIN #
#########

for epoch in range(epochs):

    model.train()
    print(f"Epoch: {epoch}")
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in progress_bar:  
        # get the inputs;
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        # Update progress bar
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.6f}"})

    model.eval()
    bg_jaccard_scores = []
    target_jaccard_scores = []
    with torch.no_grad():
        for batch in valid_dataloader:
            # get the inputs;
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            # Upsample the predicted mask to match the label size
            upsampled_logits = F.interpolate(logits.float(), size=labels.shape[-2:], mode="nearest")

            # Calculate Jaccard score (IoU) for both classes
            jaccard = jaccard_index(upsampled_logits, labels, task="multiclass", num_classes=2, average= 'none')
            bg_jaccard_scores.append(jaccard[0])
            target_jaccard_scores.append(jaccard[1])

    avg_bg_jaccard = sum(bg_jaccard_scores) / len(bg_jaccard_scores)
    avg_target_jaccard = sum(target_jaccard_scores) / len(target_jaccard_scores)
    wandb.log({"background_jaccard": avg_bg_jaccard})
    wandb.log({"target_jaccard": avg_target_jaccard})
    print(f"Epoch {epoch}, Average Background Jaccard Score: {avg_bg_jaccard:.4f}, Target Class Jaccard Score: {avg_target_jaccard:.4f}")
    
    # # Early stopping check based on target Jaccard score
    # if avg_jaccard > best_jaccard:
    #     best_jaccard = avg_jaccard
    #     epochs_no_improve = 0
    #     # Save the best model
    #     torch.save(model.state_dict(), 'best_model.pth')
    # else:
    #     epochs_no_improve += 1
    #     if epochs_no_improve == patience:
    #         print(f"Early stopping triggered. No improvement in target Jaccard score for {patience} epochs.")
    #         break
    
artifact = wandb.Artifact('finetuned_segformer', type='model')
artifact.add_file('best_model.pth')
run.log_artifact(artifact)

wandb.finish()