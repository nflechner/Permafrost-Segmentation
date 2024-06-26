import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from utils import ImageDataset, SaveFeatures, filter_dataset, imshow_transform
from cnn_classifier import model_4D
from torch.autograd import Variable
from skimage.transform import resize
from skimage.io import imshow
import wandb
import torch.optim.lr_scheduler as lr_scheduler
from skimage.segmentation import mark_boundaries, find_boundaries
from pysnic.algorithms.snic import snic
from torchmetrics.classification import MulticlassJaccardIndex


class Pseudomasks():
    def __init__(self):

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_mask(self, im, gt, save_plot: bool, cam_threshold_factor, overlap_threshold, snic_seeds, snic_compactness):

        #get the last convolution
        sf = SaveFeatures(self.model.features[-4])
        im = Variable(im).to(self.device)
        outputs = self.model(im).to(self.device)

        # generate CAM
        sf.remove()
        arr = sf.features.cpu().detach()#.numpy()

        pals_acts = torch.nn.functional.interpolate(
                                            arr[:,1,:,:].unsqueeze(1), 
                                            scale_factor = im.shape[3]/arr.shape[3], 
                                            mode='bilinear').cpu().detach()
        activation_threshold = (pals_acts.mean() + torch.std(pals_acts)) * cam_threshold_factor
        pixels_activated = torch.where(torch.Tensor(pals_acts) > activation_threshold.cpu(), 1, 0).squeeze(0).permute(1,2,0).numpy()

        # Plot image with CAM
        cpu_img = im.squeeze().cpu().detach().permute(1,2,0).long().numpy()
        superpixels = np.array(snic(cpu_img, snic_seeds, snic_compactness)[0])
        pseudomask = self.create_superpixel_mask(superpixels, pixels_activated.squeeze(), threshold=overlap_threshold)

        if save_plot: self.generate_plot(cpu_img, pals_acts, im, pixels_activated, superpixels, pseudomask, gt)

        return pseudomask

    def generate_plot(self, cpu_img, pals_acts, im, pixels_activated, superpixels, pseudomask, gt):

        # Plotting fucntion to visualize the pseudomask generation process (and intermediates)

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6, figsize = (30,6))

        ax1.imshow(cpu_img[:,:,:3])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f'original image')

        ax2.imshow(cpu_img[:,:,:3])
        ax2.imshow(pals_acts.view(im.shape[3], im.shape[3], 1), alpha=.4, cmap='jet')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title('CAM')

        ax3.imshow(pixels_activated)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title('Activated cells')

        ax4.imshow(mark_boundaries(cpu_img[:,:,:3], superpixels)) # TODO change so image is actually plotted.
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title('Superpixels')

        ax5.imshow(pseudomask)
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.set_title('Pseudomask')

        ax6.imshow(gt.squeeze(0).permute(1,2,0).long().numpy())
        ax6.set_xticks([])
        ax6.set_yticks([])
        ax6.set_title('Ground Truth')

        plt.tight_layout()
        wandb.log({'pseudomask': fig})

    def calc_metrics(self, pseudomask, gt):
        # Jaccard index (aka Intersection over Union - IoU) is the most common semantic seg metric
        metric = MulticlassJaccardIndex(num_classes=2)
        jaccard = metric(pseudomask, gt)
        return jaccard

    def model_from_artifact(self, run_id, artifact):
        # if loading the model from a wandb artifact

        run = wandb.init(project= 'VGG_CAMs', id= run_id, resume = 'must')
        artifact = run.use_artifact(f'nadjaflechner/VGG_CAMs/model:{artifact}', type='model')
        artifact_dir = artifact.download()
        state_dict = torch.load(f"{artifact_dir}/model.pth")
        model = model_4D()
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
            
    def model_from_dict(self, state_dict):
        # if loading the model from a wandb artifact
        model = model_4D()
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def create_superpixel_mask(self, superpixels, binary_mask, threshold):
        # Get the unique superpixel labels
        unique_labels = np.unique(superpixels)

        # Create a dictionary to store the overlap percentage for each superpixel
        overlap_dict = {}

        # Iterate over each superpixel
        for label in unique_labels:
            # Create a mask for the current superpixel
            superpixel_mask = (superpixels == label)

            # Count the number of pixels in the superpixel
            superpixel_size = np.sum(superpixel_mask)

            # Count the number of pixels in the superpixel that overlap with the binary mask
            overlap_count = np.sum(superpixel_mask & binary_mask)

            # Calculate the overlap percentage
            overlap_percentage = overlap_count / superpixel_size

            # Store the overlap percentage in the dictionary
            overlap_dict[label] = overlap_percentage

        # Create a new binary mask based on the overlap threshold
        new_binary_mask = np.zeros_like(binary_mask, dtype=bool)

        # Iterate over each superpixel again
        for label in unique_labels:
            # Check if the overlap percentage is greater than the threshold
            if overlap_dict[label] > threshold:
                # Set the pixels belonging to the superpixel to 1 in the new binary mask
                new_binary_mask[superpixels == label] = 1

        return new_binary_mask
