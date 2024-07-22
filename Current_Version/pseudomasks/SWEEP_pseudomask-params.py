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

##############################
# hardcoded constant configs # 
##############################

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
min_palsa_positive_samples = "TBD" #TODO fill in after hyperparam gridsearch
weight_decay = "TBD" #TODO fill in after hyperparam gridsearch
lr = "TBD" #TODO fill in after hyperparam gridsearch
lr_gamma = "TBD" #TODO fill in after hyperparam gridsearch
augment = "TBD" #TODO fill in after hyperparam gridsearch

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

########################
# define sweep configs #
########################

sweep_configuration = {
    "method": "bayes",
    "name": "pseudomask_params_sweep",
    "metric": {"goal": "maximize", "name": "test_mean_jaccard"},
    "parameters": {
        "cam_threshold_factor": {"max": 2, "min": 0.5},
        "overlap_threshold": {"max": 0.9, "min": 0.01},
        "snic_seeds": {"values": [100,200,500,1000]},
        "snic_compactness": {"values": [5,10,15,20]},
        "std_from_mean": {"values": [0,0.5,1,1.5,2,2.5]}
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="VGG_CAMs")

##############################
# define train/test function #
##############################

def train_test_model():

    cam_threshold_factor = wandb.config.cam_threshold_factor
    overlap_threshold = wandb.config.overlap_threshold
    snic_seeds = wandb.config.snic_seeds
    snic_compactness = wandb.config.snic_compactness
    std_from_mean = wandb.config.std_from_mean

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
            tags=['PseudomaskParamSearch']
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
                                    lr, weight_decay, lr_gamma, num_epochs)
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

# Start sweep
wandb.agent(sweep_id, function = train_test_model, count = 50)