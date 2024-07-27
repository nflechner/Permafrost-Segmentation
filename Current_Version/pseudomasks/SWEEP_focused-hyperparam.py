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
n_samples = 13980
batch_size = 20
num_epochs = 20
finetune = False
im_size = 200
low_pals_in_val = False
normalize = True
augment = True
depth_layer = 'hs'

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
    "name": "End2EndGridsearch",
    "metric": {"goal": "maximize", "name": "test_jaccard_palsa"},
    "parameters": {
        "min_palsa_positive_samples": {"max": 7.0, "min": 2.0},
        "weight_decay": {"max": 0.1, "min": 0.01},
        "lr": {"max": 0.00001, "min": 0.000001},
        "lr_gamma": {"max": 1.0, "min": 0.5},
        "cam_threshold_factor": {"max": 3, "min": 0.3},
        "overlap_threshold": {"max": 0.9, "min": 0.01},
        "snic_seeds": {"values": [25,40,75,100,200,500,1000]},
        "snic_compactness": {"values": [1,5,10,18,23,29,40]},
        "std_from_mean": {"max": 2.5, "min": 0}
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="VGG_CAMs")

##############################
# define train/test function #
##############################

def train_test_model():

    run = wandb.init(
        # Track hyperparameters and run metadata
        config={
            # "learning_rate": lr,
            # "lr_gamma": lr_gamma,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "n_samples": n_samples,
            "finetune": finetune,
            # "weight_decay": weight_decay,
            "im_size": im_size,
            # "min_palsa_positive_samples": min_palsa_positive_samples,
            "augment": augment,
            "normalize": normalize,
            "low_pals_in_val": low_pals_in_val,
            "depth_layer": depth_layer
            # "cam_threshold_factor": cam_threshold_factor,
            # "overlap_threshold": overlap_threshold,
            # "snic_seeds": snic_seeds,
            # "snic_compactness": snic_compactness,
            # "std_from_mean": std_from_mean
            },
            tags=['End2EndGridsearch']
    )

    min_palsa_positive_samples = wandb.config.min_palsa_positive_samples
    weight_decay = wandb.config.weight_decay
    lr = wandb.config.lr
    lr_gamma = wandb.config.lr_gamma
    cam_threshold_factor = wandb.config.cam_threshold_factor
    overlap_threshold = wandb.config.overlap_threshold
    snic_seeds = wandb.config.snic_seeds
    snic_compactness = wandb.config.snic_compactness
    std_from_mean = wandb.config.std_from_mean

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
wandb.agent(sweep_id, function = train_test_model, count = 100)