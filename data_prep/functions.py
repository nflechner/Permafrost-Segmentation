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
from shapely.geometry import box

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

    def filter_rutor(self):
        minx, miny, maxx, maxy = self.img.bounds
        img_rutor = self.rutor.cx[minx:maxx, miny:maxy] # coordinates derived manually from plotting img
        return img_rutor

    def crop_rutor(self):
        # Load the TIF file
        tif_data = self.img.read()
        tif_meta = self.img.meta

        cropped_tifs_percentages = {}
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
            cropped_tifs_percentages[f"{self.img_name_code}_crop_{idx}"] = percentage

        return cropped_tifs_percentages
    
    def generate_geoseries(self, bounds, crs):

        # height and width of new squares 
        square_dims = 200 # 200x200 pixels is 100x100 meters

        # Calculate the number of segments in each dimension (tif width // desired width in pixels!)
        segments_x = 10000 // square_dims
        segments_y = 10000 // square_dims

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
                polygon = box(left, bottom, right, top)

                # Append the polygon to the list
                polygons.append(polygon)

        # Create a GeoSeries from the list of polygons
        all_rutor = gpd.GeoSeries(polygons, crs=crs)
        return all_rutor

    def crop_negatives(self):

        num_to_sample = len(self.img_rutor)
        all_rutor = 


        # within the same images as the 
        