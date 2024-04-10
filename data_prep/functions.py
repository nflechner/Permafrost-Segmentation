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
    
    #BELOW IS JUST A PLACEHOLDER:
    return ["758_66_55_2018.tif"]

class Crop_tif():
    """
    In: tif image to be cropped, and whole extent of 100x100 rutor
    Returns: directory of one cropped tif per 100x100 ruta.
    """

    def __init__(self, img_name_code, img_path, rutor_path, destination_path):

        self.img_name_code = img_name_code

        self.img_path = img_path
        self.img = rasterio.open(img_path)

        self.rutor_path = rutor_path
        self.rutor = gpd.read_file(rutor_path)
        self.img_rutor = self.filter_rutor()

        self.destination_path = destination_path
        self.cropped_tifs_percentages = {}
        self.cropped_tifs = self.crop_tif()

    def filter_rutor(self):
        minx, miny, maxx, maxy = self.img.bounds
        img_rutor = self.rutor.cx[minx:maxx, miny:maxy] # coordinates derived manually from plotting img
        return img_rutor

    def crop_tif(self):
        # Load the TIF file
        tif_data = self.img.read()
        tif_meta = self.img.meta

        # Iterate over each polygon in the GeoDataFrame
        for idx, percentage, polygon in zip(self.img_rutor.index, self.img_rutor.PALS, self.img_rutor.geometry):
            # Crop the TIF file using the polygon
            cropped_data, cropped_transform = mask(self.img, [polygon], crop=True)

            # Update the metadata for the cropped TIF
            cropped_meta = tif_meta.copy()
            cropped_meta.update({"driver": "GTiff",
                                "height": cropped_data.shape[1],
                                "width": cropped_data.shape[2],
                                "transform": cropped_transform})

            # Save the cropped TIF file with a unique name
            output_path = os.path.join(self.destination_path, f"{self.img_name_code}_crop_{idx}.tif") # CHANGE THIS NAMING? 
            with rasterio.open(output_path, "w", **cropped_meta) as dest:
                dest.write(cropped_data)

            # Write the corresponding percentage to a dictionary as label 
            self.cropped_tifs_percentages[f"{self.img_name_code}_crop_{idx}"] = percentage
        