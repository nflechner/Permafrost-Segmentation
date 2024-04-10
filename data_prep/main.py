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


##################
## load configs ##
##################

config_path = os.path.join(os.getcwd(), 'configs.json')
with open(config_path, 'r') as config_file:
    configs = json.load(config_file)

config_paths = configs.get('paths', {})
original_tif_dir = config_paths.get('original_tif_dir')

##########
## code ##
##########

