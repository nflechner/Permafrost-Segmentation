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
from shapely.geometry import box, Polygon
import logging

##########
## code ##
##########

def tif_from_ruta(ruta_geometry):
    minx_ruta = ruta_geometry.bounds[0]
    miny_ruta = ruta_geometry.bounds[1]

    miny = str(miny_ruta)[0:3]
    minx = str(minx_ruta)[0:2]
    km_siffran_y = 0 if int(str(miny_ruta)[3]) < 5 else 5
    km_siffran_x = 0 if int(str(minx_ruta)[2]) < 5 else 5
    year = 2018

    filename = f"{miny}_{minx}_{km_siffran_y}{km_siffran_x}_{year}.tif"
    return filename


def filter_imgs(all_rutor_path, original_tif_dir):
    all_rutor = gpd.read_file(all_rutor_path)
    all_rutor['in_tif'] = all_rutor['geometry'].map(tif_from_ruta)
    uniques = all_rutor.in_tif.unique()

    dir_files = os.listdir(original_tif_dir)
    only_tifs = [filename for filename in dir_files if filename[-4:] == ".tif"]

    # check that all uniques are in only tifs
    if not (set(list(uniques)).issubset(set(only_tifs))):
        # logger.WARN(f"at least one tif name generated from all_rutor was not found in the directory: {original_tif_dir}")
        print(f"at least one tif name generated from all_rutor was not found in the directory: {original_tif_dir}")

    intersection = list(set(uniques) & set(only_tifs))

    return intersection

class Crop_tif_varsize():
    """
    In: tif image to be cropped, and whole extent of 100x100 rutor
    Returns: directory of one cropped tif per 100x100 ruta.
    """

    def __init__(self, img_name_code, img_path, rutor_path, destination_path, dims, logger):

        self.img_name_code = img_name_code
        self.dimensions = dims
        self.destination_path = destination_path
        self.logger = logger
        self.img_path = img_path
        self.rutor_path = rutor_path

        self.img = rasterio.open(img_path)
        self.filtered_rutor = self.filter_rutor()

    def forward(self):

        # generate all possible polygons in the image of dim x dim 
        generated_polygons_all = self.generate_geoseries(self.img.bounds, self.img.crs, self.dimensions) 
        generated_polygons_palsa = self.palsa_polygons(generated_polygons_all)
        positive_labels = self.crop_palsa_imgs(generated_polygons_palsa)
        negative_labels = self.crop_negatives(generated_polygons_all, generated_polygons_palsa)
        all_labels = positive_labels | negative_labels
        self.img.close()
        return all_labels

    def filter_rutor(self):
        # Find which 100x100 squares overlap with the current TIF
        rutor = gpd.read_file(self.rutor_path)
        image_polygon = box(*self.img.bounds)
        cropped_polygons = rutor[rutor.geometry.apply(lambda x: x.intersection(image_polygon).equals(x))]

        return cropped_polygons
    
    def new_palsa_percentage(self, big_ruta, joined_df):
        contained_rutor = joined_df.loc[joined_df['name'] == big_ruta]
        total_pals_percentage = contained_rutor['PALS'].sum()
        percentage_factor = self.dimensions **2 / 10000 
        palsa_percentage = total_pals_percentage / percentage_factor
        return palsa_percentage
    
    def palsa_polygons(self, generated_polygons_all):

        # if 100x100 meter is used, the original rutor are used
        if self.dimensions == 100:
            return self.filtered_rutor

        # if not 100x100, find which polygons have palsa rutor in them
        d = {'name': [i for i in range(len(generated_polygons_all))]}
        generated_polygons_all_df = gpd.GeoDataFrame(d, geometry = generated_polygons_all, crs=generated_polygons_all.crs)

        # Perform a spatial join between generated_polygons_all and filtered_rutor 
        joined_df = gpd.sjoin(generated_polygons_all_df, self.filtered_rutor, how='inner', op = 'contains')
        covering_polygons_index = joined_df.index.unique() # find uniques
        result_df = generated_polygons_all_df.loc[covering_polygons_index] # select polygons that cover a smaller polygon

        # Generate palsa column in the resulting big ruta dataframe
        result_df['PALS'] = result_df['name'].apply(lambda x: self.new_palsa_percentage(x, joined_df))

        return result_df

    def generate_geoseries(self, bounds, crs, dims):

        """
        Generates all dim x dim polygons present in a TIF.
        """

        # height and width of new squares 
        square_dims = dims # 100x100 meters

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
        return gpd.GeoSeries(polygons, crs=crs)

    def crop_palsa_imgs(self, palsa_rutor):

        """
        Crop TIF according to the polygons containing palsa. 
        """

        cropped_tifs_percentages = {}
        # Iterate over each polygon in the GeoDataFrame
        for idx, percentage, polygon in zip(palsa_rutor.index, palsa_rutor.PALS, palsa_rutor.geometry):
            # Crop the TIF file using the polygon
            cropped_data, cropped_transform = mask(self.img, [polygon], crop=True)

            # Update the metadata for the cropped TIF
            cropped_meta = self.img.meta.copy()
            cropped_meta.update({"driver": "GTiff",
                                "height": cropped_data.shape[1],
                                "width": cropped_data.shape[2],
                                "transform": cropped_transform})

            # Save the cropped TIF file with a unique name
            output_path = os.path.join(self.destination_path, f"{self.img_name_code}_crop_hs_{idx}.tif") # CHANGE THIS NAMING? 
            with rasterio.open(output_path, "w", **cropped_meta) as dest:
                dest.write(cropped_data)

            # Write the corresponding percentage to a dictionary as label 
            cropped_tifs_percentages[f"{self.img_name_code}_crop_{idx}"] = percentage

        return cropped_tifs_percentages

    def crop_negatives(self, generated_polygons_all, generated_polygons_palsa):

        """
        Generates negative samples. Equal amount of negative as positive samples are
        taken from each image such that the final dataset is 50/50 positive and negative. 

            1) split the whole TIF into 100x100m polygons.
            2) filter out the areas containing palsa (positive samples)
            3) randomly sample as many negative samples as positive samples from that image
            4) crop the TIF according to the sampled areas and write locally

        """

        # filter out the squares with palsa 
        positives_mask = ~generated_polygons_all.isin(generated_polygons_palsa.geometry)
        all_negatives = generated_polygons_all[positives_mask]

        # randomly sample 
        sample_size = int(len(generated_polygons_palsa)) # based on number of positive samples 
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
            output_path = os.path.join(self.destination_path, f"{self.img_name_code}_neg_crop_hs_{idx}.tif") # CHANGE THIS NAMING? 
            with rasterio.open(output_path, "w", **cropped_meta) as dest:
                dest.write(cropped_data)

            # Write the corresponding percentage to a dictionary as label 
            cropped_tifs_percentages[f"{self.img_name_code}_neg_crop_{idx}"] = 0

        return cropped_tifs_percentages
                
