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
from shapely.geometry import box, Polygon
import logging

##########
## code ##
##########

def file_naming_logic(img):
    "USE THIS LOGIC IN filter_imgs FUNCTION. MAYBE USE IT THE OTHER WAY AROUND"

    miny = str(img.bounds.bottom)[0:3]
    minx = str(img.bounds.left)[0:2]
    km_siffran_y = 0 if int(str(img.bounds.bottom)[3]) < 5 else 5
    km_siffran_x = 0 if int(str(img.bounds.left)[2]) < 5 else 5
    year = 2018

    filename = f"{miny}_{minx}_{km_siffran_y}{km_siffran_x}_{year}.tif"
    return filename

def filter_imgs(original_tif_dir):
    """
    In: directory of all tifs of sweden
    Returns: filenames of the images containing palsa
    """
    # use filenaming system to obtain only relevant files
    # don't load the files, only the list of file (names) in dir.
    
    #BELOW IS JUST A PLACEHOLDER:
    return ["758_66_55_2018.tif", "758_65_55_2018.tif", "758_66_50_2018.tif"]

class Crop_tif():
    """
    In: tif image to be cropped, and whole extent of 100x100 rutor
    Returns: directory of one cropped tif per 100x100 ruta.
    """

    def __init__(self, img_name_code, img_path, rutor_path, destination_path, logger):

        self.img_name_code = img_name_code

        self.img_path = img_path
        self.img = rasterio.open(img_path)

        self.rutor_path = rutor_path
        self.rutor = gpd.read_file(rutor_path)
        self.img_rutor = self.filter_rutor()

        self.destination_path = destination_path

        self.logger = logger

    def filter_rutor(self):

        """
        Find which 100x100 squares overlap with the current TIF
        """

        minx, miny, maxx, maxy = self.img.bounds
        img_rutor = self.rutor.cx[minx:maxx, miny:maxy] # coordinates derived manually from plotting img
        return img_rutor

    def crop_rutor(self):

        """
        Crop TIF according to the polygons containing palsa. 
        """

        cropped_tifs_percentages = {}
        # Iterate over each polygon in the GeoDataFrame
        for idx, percentage, polygon in zip(self.img_rutor.index, self.img_rutor.PALS, self.img_rutor.geometry):
            # Crop the TIF file using the polygon
            cropped_data, cropped_transform = mask(self.img, [polygon], crop=True)

            # Update the metadata for the cropped TIF
            cropped_meta = self.img.meta.copy()
            cropped_meta.update({"driver": "GTiff",
                                "height": cropped_data.shape[1],
                                "width": cropped_data.shape[2],
                                "transform": cropped_transform})

            # Save the cropped TIF file with a unique name
            output_path = os.path.join(self.destination_path, f"{self.img_name_code}_crop_{idx}.tif") # CHANGE THIS NAMING? 
            with rasterio.open(output_path, "w", **cropped_meta) as dest:
                dest.write(cropped_data)

            # Write the corresponding percentage to a dictionary as label 
            cropped_tifs_percentages[f"{self.img_name_code}_crop_{idx}"] = percentage

        return cropped_tifs_percentages
    
    def generate_geoseries(self, bounds, crs):

        """
        Generates all 100x100m polygons present in a TIF.
        Enables the negative sampling from the image. 
        """

        # height and width of new squares 
        square_dims = 100 # 100x100 meters

        # Calculate the number of segments in each dimension (tif width // desired width in pixels!)
        segments_x = 5000 // square_dims
        segments_y = 5000 // square_dims

        # Create an empty list to store the polygons
        polygons = []

        # Iterate over the segments
        for i in range(segments_y):
            for j in range(segments_x):
                # Calculate the coordinates of the segment
                left = bounds.left + j * square_dims
                bottom = bounds.bottom + i * square_dims
                right = left + square_dims
                top = bottom + square_dims

                # Create a polygon for the segment
                polygon = Polygon([(right, bottom), (left, bottom), (left, top), (right, top), (right, bottom)])

                # Append the polygon to the list
                polygons.append(polygon)

        # Create a GeoSeries from the list of polygons
        all_rutor = gpd.GeoSeries(polygons, crs=crs)
        return all_rutor

    def crop_negatives(self):

        """
        Generates negative samples. Equal amount of negative as positive samples are
        taken from each image such that the final dataset is 50/50 positive and negative. 

            1) split the whole TIF into 100x100m polygons.
            2) filter out the areas containing palsa (positive samples)
            3) randomly sample as many negative samples as positive samples from that image
            4) crop the TIF according to the sampled areas and write locally

        """

        # generate polygon for all 100x100m patches in the tif
        all_rutor = self.generate_geoseries(self.img.bounds, self.img.crs)

        # filter out the squares with palsa 
        positives_mask = ~all_rutor.isin(self.img_rutor.geometry)
        all_negatives = all_rutor[positives_mask]

        # randomly sample 
        sample_size = int(len(self.img_rutor)) # based on number of positive samples 
        if sample_size <= len(all_negatives): # default case
            negative_samples = all_negatives.sample(n=sample_size) # sample randomly
        else:
            self.logger.info('Exception occurred! Number of positive samples > 1/2 image. Training set now contains fewer negative than positive samples.')
            negative_samples = all_negatives

        cropped_tifs_percentages = {}
        # Iterate over each polygon in the GeoDataFrame
        for idx, polygon in enumerate(negative_samples.geometry):
            # Crop the TIF file using the polygon
            cropped_data, cropped_transform = mask(self.img, [polygon], crop=True)

            # Update the metadata for the cropped TIF
            cropped_meta = self.img.meta.copy()
            cropped_meta.update({"driver": "GTiff",
                                "height": cropped_data.shape[1],
                                "width": cropped_data.shape[2],
                                "transform": cropped_transform})

            # Save the cropped TIF file with a unique name
            output_path = os.path.join(self.destination_path, f"{self.img_name_code}_neg_crop_{idx}.tif") # CHANGE THIS NAMING? 
            with rasterio.open(output_path, "w", **cropped_meta) as dest:
                dest.write(cropped_data)

            # Write the corresponding percentage to a dictionary as label 
            cropped_tifs_percentages[f"{self.img_name_code}_neg_crop_{idx}"] = 0

        return cropped_tifs_percentages
                