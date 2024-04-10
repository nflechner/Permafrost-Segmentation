#############
## imports ##
#############

# libraries 
import geopandas as gpd
import numpy as np 
import pandas
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.mask import mask
import os
import json

# functions 
from functions import filter_imgs, Crop_tif

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

print(palsa_shapefile_path)
print(save_crops_dir)
print(original_tif_dir)

##########
## code ##
##########

# extract tif file names which contain palsa
palsa_tifs = filter_imgs(original_tif_dir) # returns a list of filenames to be cropped

# load palsa shape path
for img_name in palsa_tifs:
    img_name_code = img_name.split('.')[0]
    img_path = os.path.join(original_tif_dir, img_name)
    cropping = Crop_tif(img_name_code, img_path, palsa_shapefile_path, save_crops_dir)
    percentages = cropping.cropped_tifs_percentages
    print(percentages)


