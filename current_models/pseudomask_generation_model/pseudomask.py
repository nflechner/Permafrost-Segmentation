import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from pysnic.algorithms.snic import snic
from skimage.segmentation import mark_boundaries
from torch.autograd import Variable
from torchmetrics.classification import MulticlassJaccardIndex

from cnn_classifier import model_4D
from utils import SaveFeatures


class Pseudomasks():
    def __init__(self, test_loader, cam_threshold_factor, overlap_threshold,
                 snic_seeds, snic_compactness, finetuned):

        self.test_loader = test_loader
        self.cam_threshold_factor = cam_threshold_factor
        self.overlap_threshold= overlap_threshold
        self.snic_seeds = snic_seeds
        self.snic_compactness = snic_compactness
        self.finetuned = finetuned
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def init_model(self):
        model = model_4D()
        if self.finetuned:
            model.classifier = nn.Sequential(
                nn.Conv2d(2, 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(2, 2, kernel_size=3, padding=1))
        model.to(self.device)
        return model

    def model_from_artifact(self, run_id, artifact):
        # if loading the model from a wandb artifact

        run = wandb.init(project= 'VGG_CAMs', id= run_id, resume = 'must')
        artifact = run.use_artifact(f'nadjaflechner/VGG_CAMs/model:{artifact}', type='model')
        artifact_dir = artifact.download()
        state_dict = torch.load(f"{artifact_dir}/model.pth")
        model = self.init_model()
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

    def model_from_dict(self, state_dict):
        # if loading the model from a wandb artifact
        model = self.init_model()
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

    def test_loop(self, test_loader):

        running_jaccard = 0
        for i in range(len(test_loader.dataset)):
            im, lab, perc_label, gt_mask = next(iter(test_loader))
            if not lab == 0:  # currently not yet comparing negative samples
                pseudomask = self.generate_mask(im, gt_mask, save_plot=True)
                # calculate metrics to evaluate model on test set
                generated_mask = torch.Tensor(pseudomask).int().view(400,400).to(self.device)
                groundtruth_mask = torch.Tensor(gt_mask).int().view(400,400).to(self.device)
                metrics = self.calc_metrics(generated_mask, groundtruth_mask)
                running_jaccard += metrics
        wandb.log({"test_mean_jaccard": metrics / len(test_loader.dataset)})

    def generate_mask(self, im, gt, save_plot: bool):

        #get the last convolution
        if not self.finetuned:
            sf = SaveFeatures(self.model.features[-4])
        else: sf = SaveFeatures(self.model.classifier[-1])
        im = Variable(im).to(self.device)
        outputs = self.model(im).to(self.device)

        # generate CAM
        sf.remove()
        arr = sf.features.cpu().detach()#.numpy()

        pals_acts = torch.nn.functional.interpolate(
                                            arr[:,1,:,:].unsqueeze(1),
                                            scale_factor = im.shape[3]/arr.shape[3],
                                            mode='bilinear').cpu().detach()
        activation_threshold = (pals_acts.mean() + pals_acts.std()) * torch.tensor(self.cam_threshold_factor)
        pixels_activated = torch.where(torch.Tensor(pals_acts) > activation_threshold.cpu(), 1, 0).squeeze(0).permute(1,2,0).numpy()

        # Plot image with CAM
        cpu_img = im.squeeze().cpu().detach().permute(1,2,0).long().numpy()
        superpixels = np.array(snic(cpu_img, int(self.snic_seeds), self.snic_compactness)[0])
        pseudomask = self.create_superpixel_mask(superpixels, pixels_activated.squeeze(), threshold=self.overlap_threshold)

        if save_plot: self.generate_plot(cpu_img, pals_acts, im, pixels_activated, superpixels, pseudomask, gt)

        return pseudomask

    def generate_plot(self, cpu_img, pals_acts, im, pixels_activated, superpixels, pseudomask, gt):

        # Plotting fucntion to visualize the pseudomask generation process (and intermediates)

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6, figsize = (30,6))

        ax1.imshow(cpu_img[:,:,:3])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('original image')

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
        metric = MulticlassJaccardIndex(num_classes=2).to(self.device)
        jaccard = metric(pseudomask, gt)
        return jaccard

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
