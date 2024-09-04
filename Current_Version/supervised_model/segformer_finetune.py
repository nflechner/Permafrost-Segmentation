import torch
from torch import nn
from torchmetrics.functional import jaccard_index
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

    def __init__(self, root_dir, image_processor):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegformerImageProcessor): image processor to prepare images + segmentation maps.
        """
        self.root_dir = root_dir
        self.image_processor = image_processor

        self.img_dir = os.path.join(self.root_dir, "images")
        self.ann_dir = os.path.join(self.root_dir, "masks")
        
        # Get all image filenames without extension
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f.endswith('.jpg')]

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


root_dir = "/root/Permafrost-Segmentation/Supervised_dataset"
image_processor = SegformerImageProcessor(
    image_mean = [74.90, 85.26, 80.06], # use mean calculated over our dataset
    image_std = [15.05, 13.88, 12.01], # use std calculated over our dataset
    do_reduce_labels=False
    )

# Create the full dataset
full_dataset = SemanticSegmentationDataset(root_dir, image_processor)

# Split the dataset into 85% train and 15% validation
total_size = len(full_dataset)
train_size = int(0.85 * total_size)
valid_size = total_size - train_size

train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64)

# define model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0", 
    num_labels=1# since we treat '0' as a background, the only class is palsa.
) 

# Freeze encoder layers
for param in model.segformer.encoder.parameters():
    param.requires_grad = False

# Optionally, unfreeze the last few layers of the encoder
# Adjust the number of unfrozen blocks as needed
num_unfrozen_blocks = 2
for i in range(len(model.segformer.encoder.block) - num_unfrozen_blocks, len(model.segformer.encoder.block)):
    for param in model.segformer.encoder.block[i].parameters():
        param.requires_grad = True


epochs = 20
lr = 0.00006
warmup_steps = 100  # Adjust this value as needed

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# define scheduler
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Move optimizer to GPU (possibly unneccessary)
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

# Early stopping parameters
patience = 5
best_jaccard = 0
epochs_no_improve = 0

run = wandb.init(
    # Set the project where this run will be logged
    project="Finetune_segformer",
    # Track hyperparameters and run metadata
    config={
        "epochs": 20,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "patience": patience
        }
)

model.train()
for epoch in range(epochs):
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
    jaccard_scores = []
    target_jaccard_scores = []
    with torch.no_grad():
        for batch in valid_dataloader:
            # get the inputs;
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            # Calculate Jaccard score
            # Since we only have one feature map, we can use a threshold to determine the segmentation
            predicted = (logits.squeeze(1) > 0).float()  # Threshold at 0
            upsampled_predicted = F.interpolate(predicted.unsqueeze(1), size=labels.shape[-2:], mode="nearest")

            # Calculate Jaccard score (IoU) for both classes
            jaccard = jaccard_index(upsampled_predicted.squeeze(1), labels, task="multiclass", num_classes=2)
            jaccard_scores.append(jaccard.item())

            # Calculate Jaccard score (IoU) for target class only, if not a only background image
            if len(labels.unique()) > 1:
                target_jaccard = jaccard_index(upsampled_predicted.squeeze(1), labels, task="multiclass", num_classes=2, average="none")[1]
                target_jaccard_scores.append(target_jaccard.item())

    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
    avg_target_jaccard = sum(target_jaccard_scores) / len(target_jaccard_scores)
    wandb.log({"jaccard": avg_jaccard})
    wandb.log({"target_jaccard": avg_target_jaccard})
    print(f"Epoch {epoch}, Average Jaccard Score: {avg_jaccard:.4f}, Target Class Jaccard Score: {avg_target_jaccard:.4f}")
    
    # Early stopping check based on target Jaccard score
    if avg_jaccard > best_jaccard:
        best_jaccard = avg_jaccard
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping triggered. No improvement in target Jaccard score for {patience} epochs.")
            break
    
    model.train()


artifact = wandb.Artifact('finetuned_segformer', type='model')
artifact.add_file('best_model.pth')
run.log_artifact(artifact)

wandb.finish()