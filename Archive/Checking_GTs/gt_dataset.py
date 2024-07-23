import os

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

class TestSet(Dataset):
    def __init__(self, depth_layer, gt_dir, normalize):
        self.RGB_dir = os.path.join(gt_dir, 'rgb')
        self.hs_dir = os.path.join(gt_dir, 'hs')
        self.dem_dir = os.path.join(gt_dir, 'dem')
        self.groundtruth_dir = os.path.join(gt_dir, 'groundtruth_mask')
        self.depth_dir = self.hs_dir if depth_layer == "hs" else self.dem_dir
        self.labels_path = os.path.join(gt_dir, 'new_palsa_labels.csv')
        self.im_size = 200
        self.normalize = normalize

        # configure labels file.
        # only use samples where MS-Backe difference is <10%
        unfiltered_labels_df = pd.read_csv(self.labels_path, index_col=0)
        filtered_labels_df = unfiltered_labels_df.loc[
            (unfiltered_labels_df['difference']<10)]
        self.labels_df = filtered_labels_df.loc[
            (filtered_labels_df['palsa_percentage']>0)]

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.index[idx]
        RGB_img_path = os.path.join(self.RGB_dir, f"{img_name}.tif")
        hs_img_path = os.path.join(self.depth_dir, f"{img_name}.tif")

        with rasterio.open(RGB_img_path) as RGB_src:
            # Read the image data
            RGB_img = RGB_src.read()

        with rasterio.open(hs_img_path) as hs_src:
            # Read the image data
            hs_img = hs_src.read()

        label = self.labels_df.iloc[idx, 0]
        perc_label = label/100

        # grab ground truth mask
        gt_img_path = os.path.join(self.groundtruth_dir, f"{img_name}.tif")
        with rasterio.open(gt_img_path) as gt_src:
            gt_mask = gt_src.read()

        return RGB_img, hs_img, perc_label, gt_mask, img_name