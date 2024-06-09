#############
## imports ##
#############

# libraries
import json
import logging
import os

import pandas as pd

# functions
from functions import Crop_tif_varsize, filter_imgs, get_RGB_match

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
RGB_tif_dir = config_paths.get('RGB_tif_dir') # load directory with all tifs
hillshade_tif_dir = config_paths.get('hillshade_tif_dir') # load directory with all tifs
DEM_tif_dir = config_paths.get('DEM_tif_dir') # load directory with all tifs

config_img = configs.get('image_info', {})
dims = int(config_img.get('meters_per_axis'))

logger.info('Configurations were loaded')

##########
## code ##
##########

# Filter hillshade data so only those containing palsa remain
logger.info('Starting to sample relevant TIF paths...')
hillshade_filenames = filter_imgs(palsa_shapefile_path, hillshade_tif_dir) # TODO: could already filter 'only newest' here..
logger.info(f'{len(hillshade_filenames)} TIF paths have been loaded!')

# Loop over hillshade images to generate the crops.
logger.info('Starting to generate training samples from TIFs..')
labels = {}
not_found = []
for idx, hs_img_name in enumerate(hillshade_filenames):
    # grab corresponding RGB image (matching the hillshade)
    try:
        RGB_tif_name = get_RGB_match(hs_img_name, RGB_tif_dir)
        RGB_img_name_code = RGB_tif_name.split('.')[0]
        RGB_img_path = os.path.join(RGB_tif_dir, RGB_tif_name)

        hs_img_name_code = hs_img_name.split('.')[0]
        hs_img_path = os.path.join(hillshade_tif_dir, hs_img_name)

        DEM_img_name_code = hs_img_name.split('.')[0]
        DEM_img_path = os.path.join(DEM_tif_dir, hs_img_name)

        cropping = Crop_tif_varsize(RGB_img_name_code, RGB_img_path, hs_img_name_code,
                                    hs_img_path, DEM_img_name_code, DEM_img_path,
                                    palsa_shapefile_path, save_crops_dir, dims, logger)

        # Run the cropping script
        new_labels = cropping.forward()
        labels = labels | new_labels
        logger.info(f'Generated training samples from image {idx+1}/{len(hillshade_filenames)}')
    except:
        logger.info(f'RGB or DEM match for {hs_img_name} not found')
        not_found.append(hs_img_name)

print(f'The following images had no rgb match: \n {not_found}')
print(f'number of images where script failed: \n {len(not_found)}')

label_df = pd.DataFrame.from_dict(labels, orient='index', columns = ['palsa_percentage'])
label_df.to_csv(os.path.join(save_crops_dir, "palsa_labels.csv"))
