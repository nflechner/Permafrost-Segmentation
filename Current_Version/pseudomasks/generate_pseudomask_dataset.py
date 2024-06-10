############
# Imports #
############

import json
import os

import wandb
import pandas as pd
import rasterio
from torch.utils.data import DataLoader

from model.pseudomask import Pseudomasks
from utils.data_modules import ImageDataset, TestSet, filter_dataset

##################
## load configs ##
##################

# use this path when using vs code debugger.
# config_path = os.path.join('/home/nadjaflechner/palsa_seg/current_models/pseudomask_generation_model', 'configs.json')

config_path = os.path.join(os.getcwd(), 'pseudomasks_configs/configs.json')
with open(config_path, 'r') as config_file:
    configs = json.load(config_file)

# load paths configs dictionary
config_paths = configs.get('paths', {})
# assign paths
palsa_shapefile = config_paths.get('palsa_shapefile')
final_pseudomasks_dir = config_paths.get('final_pseudomasks_dir')
testset_dir = config_paths.get('testset')
parent_dir = config_paths.get('data')
rgb_dir = os.path.join(parent_dir, 'rgb')
hs_dir = os.path.join(parent_dir, 'hs')
dem_dir = os.path.join(parent_dir, 'dem')
labels_file = os.path.join(parent_dir, 'palsa_labels.csv')

# load model configs dictionary
config_model = configs.get('model', {})
# assign model
artifact_path = config_model.get('artifact_path')
run_id = config_model.get('run_id')
finetune = config_model.get('finetuned')

# load data configs dictionary
config_data = configs.get('data', {})
# assign data configs
n_samples = config_data.get('n_samples')
batch_size = config_data.get('batch_size')
im_size = config_data.get('im_size')
min_palsa_positive_samples = config_data.get('min_palsa_positive_samples')
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


#########################
# configure dataloaders #
#########################

train_files, val_files = filter_dataset(
    labels_file = labels_file,
    augment = augment,
    min_palsa_positive_samples = min_palsa_positive_samples,
    low_pals_in_val = False, 
    n_samples = n_samples
    )
all_files = pd.concat([train_files, val_files])

# choose depth data based on configs
depth_dir = hs_dir if depth_layer == "hs" else dem_dir
# Create the dataset and loaders for the entire dataset.
dataset = ImageDataset(depth_dir, rgb_dir, all_files, im_size, normalize)
loader = DataLoader(dataset, batch_size=1, shuffle=True)


#############################
# generate all pseudolabels #
#############################

test_set = TestSet(depth_layer, testset_dir, normalize)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)

pseudomask_generator = Pseudomasks(
    test_loader, cam_threshold_factor, overlap_threshold,
    snic_seeds, snic_compactness, finetuned = finetune
    )
pseudomask_generator.model_from_artifact(run_id, artifact_path)

for im,_,_,img_name in loader:
    pseudomask = pseudomask_generator.generate_mask(im, None, save_plot=False)
    # Update the metadata for the cropped TIF
    cropped_meta = im.meta.copy()
    cropped_meta.update({"driver": "GTiff",
                        "height": im.shape[1],
                        "width": im.shape[2]})
    
    print(type(pseudomask))
    print(pseudomask.shape)

    break
    # Save the cropped TIF file with a unique name
    output_path = os.path.join(final_pseudomasks_dir, f"{img_name}.tif")
    with rasterio.open(output_path, "w", **cropped_meta) as dest:
        dest.write(pseudomask)


##############
# finish run #
##############

wandb.finish()
