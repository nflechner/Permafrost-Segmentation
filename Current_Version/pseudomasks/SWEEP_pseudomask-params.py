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

############
# SWEEP 1 # 
############

# model to be tested
model = 'classification_model:v48'

artifact_path = f'nadjaflechner/VGG_CAMs/{model}'
testset_dir = '/Users/nadja/Documents/UU/Thesis/Data/Verified_GTs'
depth_layer = 'hs'
normalize = True
finetune = False
std_from_mean = 0

sweep_configuration = {
    "method": "random",
    "name": f"pseudomasks_{model}",
    "metric": {"goal": "maximize", "name": "test_jaccard_palsa"},
    "parameters": {
        "cam_threshold_factor": {"max": 2.2, "min": 0.3},
        "overlap_threshold": {"max": 0.7, "min": 0.01},
        "snic_seeds": {"values": [20,40,60,80,100]},
        "snic_compactness": {"max": 25, "min": 1}
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="VGG_CAMs")

test_set = TestSet(depth_layer, testset_dir, normalize)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)

pseudomask_generator = Pseudomasks(test_loader, None, None,
                                    None, None, finetune, std_from_mean)

api = wandb.Api()
artifact = api.artifact(artifact_path, type='model')
artifact_dir = artifact.download()
state_dict = torch.load(f"{artifact_dir}/model.pth", map_location=torch.device('cpu'))
pseudomask_generator.model_from_dict(state_dict)

def train_test_model():

    run = wandb.init(
        # Track hyperparameters and run metadata
        config={
            "model": model,
            "normalize": normalize
            },
            tags=['OnlyPseudomaskParams']
    )

    pseudomask_generator.cam_threshold = wandb.config.cam_threshold_factor
    pseudomask_generator.overlap_threshold = wandb.config.overlap_threshold
    pseudomask_generator.snic_seeds = wandb.config.snic_seeds
    pseudomask_generator.snic_compactness = wandb.config.snic_compactness

    pseudomask_generator.test_loop(test_loader)

# Start sweep
wandb.agent(sweep_id, function = train_test_model, count = 35)

