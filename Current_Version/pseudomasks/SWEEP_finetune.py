"""
main idea: 
- take a trained model version (use the same for every finetuning run)
- only do the finetuning (not cnn training)


""" 

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
finetune = True
im_size = 200
low_pals_in_val = False
normalize = True
augment = True
depth_layer = 'hs'
cam_threshold_factor = 0.95
overlap_threshold = 0.3
snic_seeds = 100
snic_compactness = 10
std_from_mean = 2
min_palsa_positive_samples = 6.2


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
    "name": "Finetune_sweep",
    "metric": {"goal": "maximize", "name": "test_jaccard_palsa"},
    "parameters": {
        "weight_decay": {"max": 0.1, "min": 0.01},
        "lr": {"values": [0.000001, 0.00001,0.0001,0.001]},
        "lr_gamma": {"max": 1.0, "min": 0.5},
        "finetune_threshold": {"max": 5.0, "min": 0.5},
        "num_layers_freeze": {"values": [11,14,18,21]}
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="VGG_CAMs")

##############################
# define train/test function #
##############################

def finetune_model():

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
            "min_palsa_positive_samples": min_palsa_positive_samples,
            "augment": augment,
            "normalize": normalize,
            "low_pals_in_val": low_pals_in_val,
            "depth_layer": depth_layer,
            "cam_threshold_factor": cam_threshold_factor,
            "overlap_threshold": overlap_threshold,
            "snic_seeds": snic_seeds,
            "snic_compactness": snic_compactness,
            "std_from_mean": std_from_mean
            },
            tags=['FinetuneSweep']
    )

    weight_decay = wandb.config.weight_decay
    lr = wandb.config.lr
    lr_gamma = wandb.config.lr_gamma
    finetune_threshold = wandb.config.finetune_threshold
    num_layers_freeze = wandb.config.num_layers_freeze

    # configure dataloaders #
    train_files, val_files = filter_dataset(labels_file, augment, min_palsa_positive_samples, low_pals_in_val, n_samples)
    # choose depth data based on configs
    depth_dir = hs_dir if depth_layer == "hs" else dem_dir
    # Create the datasets and data loaders
    train_dataset = ImageDataset(depth_dir, rgb_dir, train_files, im_size, normalize)
    val_dataset = ImageDataset(depth_dir, rgb_dir, val_files, im_size, normalize)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    artifact = run.use_artifact('nadjaflechner/VGG_CAMs/classification_model:v139', type='model')
    artifact_dir = artifact.download()
    state_dict = torch.load(f"{artifact_dir}/model.pth")

    # define model #
    model = model_4D()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(state_dict)
    # finetune pretrained model (overwrite best_model to evaluate)
    best_model = FinetuneLoop(model, train_loader, val_loader,
                                lr, weight_decay, lr_gamma, num_epochs, 
                                finetune_threshold, num_layers_freeze)
    # after all epochs, save the best model as an artifact to wandb
    torch.save(best_model, '/home/nadjaflechner/models/model.pth')
    artifact = wandb.Artifact('finetuned_model', type='model')
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
wandb.agent(sweep_id, function = finetune_model, count = 50)