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

########################
# define sweep configs #
########################

# Define sweep config
sweep_configuration = {
    "method": "random",
    "name": "focused_hyperparam_sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "min_palsa_positive_samples": {"max": 7, "min": 2},
        "weight_decay": {"max": 0.1, "min": 0.01},
        "lr": {"max": 0.00001, "min": 0.000001},
        "lr_gamma": {"max": 1, "min": 0.5},
        "augment": {"values": [True, False]},
    },
}

##########################
# hardcode other configs #
##########################

config_path = os.path.join(os.getcwd(), 'configs/classifier_configs.json')
with open(config_path, 'r') as config_file:
    configs = json.load(config_file)

# load hyperparams configs dictionary
config_hyperparams = configs.get('hyperparams', {})
# assign hyperparams
n_samples = 10200
batch_size = 20
num_epochs = 20
finetune = False
im_size = 200
low_pals_in_val = False
normalize = True
depth_layer = 'hs'
cam_threshold_factor = 0.95
overlap_threshold = 0.3
snic_seeds = 100
snic_compactness = 10

##################
## load paths ##
##################

# use this path when using vs code debugger.
# config_path = os.path.join('/home/nadjaflechner/palsa_seg/current_models/pseudomask_generation_model', 'configs.json')

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

#################################################################################
################ WAS DONE UNTIL HERE, CONTINUE BELOW

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
        tags=['FinalGridsearch']
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