############
# Imports #
############

import json
import os

import torch
import wandb
from torch.utils.data import DataLoader

from model.cnn_classifier import model_4D
from model.finetune import FinetuneLoop
from model.pseudomask import Pseudomasks
from model.train import ClassifierTrainLoop
from utils.data_modules import ImageDataset, TestSet, filter_dataset


##################
## load configs ##
##################

config_path = os.path.join(configs_dir, configsfile)
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

# load pseudomasks configs dictionary
config_pseudomasks = configs.get('pseudomasks', {})
# assign pseudomasks configs
cam_threshold_factor = config_pseudomasks.get('cam_threshold_factor')
overlap_threshold = config_pseudomasks.get('overlap_threshold')
snic_seeds = config_pseudomasks.get('snic_seeds')
snic_compactness = config_pseudomasks.get('snic_compactness')

##########################
# log hyperparams to w&b #
##########################

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
        "depth_layer": depth_layer,
        "cam_threshold_factor": cam_threshold_factor,
        "overlap_threshold": overlap_threshold,
        "snic_seeds": snic_seeds,
        "snic_compactness": snic_compactness
        },
        tags=['FinalGridsearchWTest']
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
    # finetune pretrained model (overwrite best_model to evaluate)
    best_model = FinetuneLoop(model, train_loader, val_loader,
                                lr, weight_decay, lr_gamma, num_epochs)
    # after all epochs, save the best model as an artifact to wandb
    torch.save(best_model, '/home/nadjaflechner/models/model.pth')
    artifact = wandb.Artifact('finetuned_model', type='model')
    artifact.add_file('/home/nadjaflechner/models/model.pth')
    run.log_artifact(artifact)


##################
# evaluate model #
##################

print('Testing model ...')
test_set = TestSet(depth_layer, testset_dir, normalize)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)

pseudomask_generator = Pseudomasks(test_loader, cam_threshold_factor, overlap_threshold,
                                    snic_seeds, snic_compactness, finetuned = finetune)
pseudomask_generator.model_from_dict(best_model)
pseudomask_generator.test_loop(test_loader)


##############
# finish run #
##############

wandb.finish()
