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

sweep_configuration = {
    "method": "grid",
    "name": "CAM_threshold",
    "metric": {"goal": "maximize", "name": "test_jaccard_palsa"},
    "parameters": {
        # "cam_threshold_factor": {"max": 0.95, "min": 0.3},
        "std_from_mean": {"values": [0.5,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3]}
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="VGG_CAMs")

##############################
# define train/test function #
##############################

def train_test_model():

    run = wandb.init(
            tags=['PseudomaskGridsearch']
    )

    # collect run configs
    cam_threshold_factor = 0
    std_from_mean = wandb.config.std_from_mean

    # define model
    model_path = "nadjaflechner/VGG_CAMs/classification_model:v61"
    api = wandb.Api()
    artifact = api.artifact(model_path, type='model')
    artifact_dir = artifact.download()
    state_dict = torch.load(f"{artifact_dir}/model.pth")

    # define dataset
    GT_dir = "/Volumes/USB/Ground_truth/Optimize_GTs"
    dataset = TestSet("hs", GT_dir, normalize = False)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers = 1)

    # Make pseudomask generator
    pseudomask_generator = Pseudomasks(test_loader, cam_threshold_factor, overlap_threshold=0,
                                        snic_seeds=0, snic_compactness=0, finetune=False, std_from_mean=std_from_mean)
    pseudomask_generator.model_from_dict(state_dict)
    pseudomask_generator.test_loop_thresholdedCAMs(test_loader)

# Start sweep
wandb.agent(sweep_id, function = train_test_model, count = 100)