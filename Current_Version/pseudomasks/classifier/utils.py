import os

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms


def filter_dataset(
    labels_file, augment, min_palsa_positive_samples, low_pals_in_val, n_samples):

    n_train = int(round(n_samples*0.8))
    n_val = int(round(n_samples*0.2))

    # labels as dataframe
    labels_df = pd.read_csv(labels_file, index_col=0)

    # remove augmented images depending on configs
    if not augment:
        labels_df = labels_df.loc[~labels_df.index.str.endswith('_aug')]

    # impose minimum palsa percentage for positive samples
    if min_palsa_positive_samples > 0:

        # find indices of samples with 0<x<threshold palsa
        drop_range = labels_df[
            (labels_df['palsa_percentage'] > 0)
            & (labels_df['palsa_percentage'] <= min_palsa_positive_samples)].index

        # remove low palsa images from train set
        train_df = labels_df.drop(drop_range).sample(n_train)

        # sample val images from unfiltered df
        if low_pals_in_val:
            val_df = labels_df.drop(train_df.index).sample(n_val)

        # sample val images from filtered df
        if not low_pals_in_val:
            val_df = labels_df.drop(drop_range)
            val_df = val_df.drop(train_df.index).sample(n_val)

    # if no minimum palsa percentage is imposed
    elif min_palsa_positive_samples == 0:
        train_df = labels_df.sample(n_train)
        val_df = labels_df.drop(train_df.index).sample(n_val)

    return train_df, val_df


class ImageDataset(Dataset):
    def __init__(self, depth_dir, RGB_dir, labels_df, im_size, normalize):
        self.RGB_dir = RGB_dir
        self.depth_dir = depth_dir
        self.labels_df = labels_df
        self.im_size = im_size
        self.normalize = normalize

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

        # convert and upsample hs image
        hs_image_array = np.array(hs_img)
        hs_image_tensor = torch.from_numpy(hs_image_array)
        hs_image_tensor = hs_image_tensor.float()
        bilinear = nn.Upsample(size=self.im_size*2, mode='bilinear')
        hs_upsampled_tensor = bilinear(hs_image_tensor.unsqueeze(0)).squeeze(0)

        # converting RGB to tensor
        RGB_image_array = np.array(RGB_img)
        RGB_image_tensor = torch.from_numpy(RGB_image_array)
        RGB_image_tensor = RGB_image_tensor.float()

        combined_tensor = torch.concatenate((RGB_image_tensor, hs_upsampled_tensor))

        if self.normalize:
            # use dataset wide calculated means and standard deviations
            if str(self.depth_dir).endswith('hs'):
                transforms.Normalize(mean=[74.90, 85.26, 80.06,179.18],
                                     std=[15.05, 13.88, 12.01,10.65])
                pass
            if str(self.depth_dir).endswith('dem'):
                transforms.Normalize(mean=[74.90, 85.26, 80.06,608.95],
                                     std=[15.05, 13.88, 12.01, 2.30])
                pass

        label = self.labels_df.iloc[idx, 0]
        binary_label = 1 if label > 0 else 0
        perc_label = label/100
        return combined_tensor, binary_label, perc_label


class TestSet(Dataset):
    def __init__(self, depth_layer, gt_dir, normalize):
        self.RGB_dir = os.path.join(gt_dir, 'rgb')
        self.hs_dir = os.path.join(gt_dir, 'hs')
        self.dem_dir = os.path.join(gt_dir, 'dem')
        self.groundtruth_dir = os.path.join(gt_dir, 'groundtruth_mask')
        self.depth_dir = self.hs_dir if depth_layer == "hs" else self.dem_dir
        self.labels_path = os.path.join(gt_dir, 'palsa_labels.csv')
        self.labels_df = pd.read_csv(self.labels_path, index_col=0)
        self.im_size = 200
        self.normalize = normalize

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

        # convert and upsample hs image
        hs_image_array = np.array(hs_img)
        hs_image_tensor = torch.from_numpy(hs_image_array)
        hs_image_tensor = hs_image_tensor.float()
        bilinear = nn.Upsample(size=self.im_size*2, mode='bilinear')
        hs_upsampled_tensor = bilinear(hs_image_tensor.unsqueeze(0)).squeeze(0)

        # converting RGB to tensor
        RGB_image_array = np.array(RGB_img)
        RGB_image_tensor = torch.from_numpy(RGB_image_array)
        RGB_image_tensor = RGB_image_tensor.float()

        combined_tensor = torch.concatenate((RGB_image_tensor, hs_upsampled_tensor))

        if self.normalize:
            # use dataset wide calculated means and standard deviations
            if str(self.depth_dir).endswith('hs'):
                transforms.Normalize(mean=[74.90, 85.26, 80.06,179.18],
                                     std=[15.05, 13.88, 12.01,10.65])
                pass
            if str(self.depth_dir).endswith('dem'):
                transforms.Normalize(mean=[74.90, 85.26, 80.06,608.95],
                                     std=[15.05, 13.88, 12.01, 2.30])
                pass

        label = self.labels_df.iloc[idx, 0]
        binary_label = 1 if label > 0 else 0
        perc_label = label/100

        # grab ground truth mask
        gt_img_path = os.path.join(self.groundtruth_dir, f"{img_name}.tif")
        with rasterio.open(gt_img_path) as gt_src:
            gt_mask = gt_src.read()

        gt_image_array = np.array(gt_mask)
        gt_image_tensor = torch.from_numpy(gt_image_array)
        gt_image_tensor = gt_image_tensor.float()
        gt_upsampled_tensor = bilinear(gt_image_tensor.unsqueeze(0)).squeeze(0)  # OUTPUT is np array (1,200,200)

        return combined_tensor, binary_label, perc_label, gt_upsampled_tensor

# https://www.fast.ai/
# fastai code snippet
class SaveFeatures():
    features=None
    def __init__(self,m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()
