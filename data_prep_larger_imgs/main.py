#############
## imports ##
#############

# libraries 
import geopandas as gpd
import numpy as np 
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.mask import mask
import os
import json
import logging

# functions 
from functions import filter_imgs, Crop_tif, Crop_tif_varsize


##################
## setup logger ##
##################

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# Setup logger
ch = logging.StreamHandler() # create console handler
ch.setLevel(logging.DEBUG) # set level to debug
formatter = logging.Formatter("%(asctime)s - %(message)s \n", "%Y-%m-%d %H:%M:%S") # create formatter
ch.setFormatter(formatter) # add formatter to ch
logger.addHandler(ch) # add ch to logger

logger.info('Imports successful')

##################
## load configs ##
##################

config_path = os.path.join(os.getcwd(), 'configs.json')
with open(config_path, 'r') as config_file:
    configs = json.load(config_file)

# load paths from configs 
config_paths = configs.get('paths', {}) 
palsa_shapefile_path = config_paths.get('palsa_shapefile_path') # load shapefile path
save_crops_dir = config_paths.get('save_crops_dir') # load directory with all tifs
original_tif_dir = config_paths.get('original_tif_dir') # load directory with all tifs

config_img = configs.get('image_info', {}) 
dims = int(config_img.get('meters_per_axis')) 

logger.info('Configurations were loaded')

##########
## code ##
##########

logger.info('Starting to sample relevant TIF paths...')

# extract tif file names which contain palsa
palsa_tifs = filter_imgs(palsa_shapefile_path, original_tif_dir) # returns a list of filenames to be cropped

logger.info(f'{len(palsa_tifs)} TIF paths have been loaded!')
logger.info('Starting to generate training samples from TIFs..')

labels = {}

# load palsa shape path
for idx, img_name in enumerate(palsa_tifs):
    img_name_code = img_name.split('.')[0]
    img_path = os.path.join(original_tif_dir, img_name)
    if dims == 100:
        cropping = Crop_tif(img_name_code, img_path, palsa_shapefile_path, save_crops_dir, logger)
        positive_labels = cropping.crop_rutor()
        negative_labels = cropping.crop_negatives()
        all_labels = positive_labels | negative_labels
        labels = labels | all_labels
        logger.info(f'Generated training samples from image {idx+1}/{len(palsa_tifs)}')
    else:
        cropping = Crop_tif_varsize(img_name_code, img_path, palsa_shapefile_path, save_crops_dir, dims, logger)
        positive_labels = cropping.crop_rutor()
        negative_labels = cropping.crop_negatives()
        all_labels = positive_labels | negative_labels
        labels = labels | all_labels
        logger.info(f'Generated training samples from image {idx+1}/{len(palsa_tifs)}')

label_df = pd.DataFrame.from_dict(labels, orient='index', columns = ['palsa_percentage'])
label_df.to_csv(os.path.join(save_crops_dir, "palsa_labels.csv"))
