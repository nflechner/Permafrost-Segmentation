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

##########
## code ##
##########

def filter_imgs(original_tif_dir):
    """
    In: directory of all tifs of sweden
    Returns: filenames of the images containing palsa
    """
    # use filenaming system to obtain only relevant files
    # don't load the files, only the list of file (names) in dir.
    pass

class Crop_tif():
    """
    In: tif image to be cropped, and whole extent of 100x100 rutor
    Returns: directory of one cropped tif per 100x100 ruta.
    """

    def __init__(self, img_path, rutor_path):

        self.img_path = img_path
        self.img = rasterio.open(img_path)

        self.rutor_path = rutor_path
        self.rutor = gpd.read_file(rutor_path)
        self.img_rutor = self.filter_rutor()

    def filter_rutor(self):
        minx, miny, maxx, maxy = self.img.bounds
        img_rutor = self.rutor.cx[minx:maxx, miny:maxy] # coordinates derived manually from plotting img
        return img_rutor

    def crop_tif(self):
        pass