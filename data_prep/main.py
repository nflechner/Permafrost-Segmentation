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
from functions import filter_imgs

##################
## load configs ##
##################

config_path = os.path.join(os.getcwd(), 'configs.json')
with open(config_path, 'r') as config_file:
    configs = json.load(config_file)

##########
## code ##
##########

# extract files containing palsa
config_paths = configs.get('paths', {}) 
original_tif_dir = config_paths.get('original_tif_dir') # load directory with all tifs
palsa_tifs = filter_imgs(original_tif_dir)

# load all 100x100m 'rutor' containing palsa
palsa_shapefile_path = config_paths.get('palsa_shapefile_path') # load shapefile path
