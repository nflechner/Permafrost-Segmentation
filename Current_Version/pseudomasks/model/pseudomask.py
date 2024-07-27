import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
import wandb
from pysnic.algorithms.snic import snic
from skimage.segmentation import mark_boundaries
from torch.autograd import Variable
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassF1Score

from model.cnn_classifier import model_4D
from utils.data_modules import SaveFeatures


class Pseudomasks():
    def __init__(self, test_loader, cam_threshold, overlap_threshold,
                 snic_seeds, snic_compactness, finetuned, std_from_mean):

        self.test_loader = test_loader
        self.cam_threshold = cam_threshold
        self.std_from_mean = std_from_mean
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

    def model_from_artifact(self, artifact_path = 'nadjaflechner/VGG_CAMs/finetuned_model:V3829'):
        # if loading the model from a wandb artifact

        run = wandb.init(project= 'VGG_CAMs')
        artifact = run.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        # state_dict = torch.load(f"{artifact_dir}/model.pth")
        # TODO: REMOVE LINE BELOW (FOR RUNNING ON MY MACBOOK)
        state_dict = torch.load(f"{artifact_dir}/model.pth", map_location=torch.device('cpu'))
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

        running_jaccard_nopalsa = 0
        running_accuracy_nopalsa = 0
        running_F1_nopalsa = 0

        running_jaccard_palsa = 0
        running_accuracy_palsa = 0
        running_F1_palsa = 0

        for im, lab, _, gt_mask in test_loader:
            pseudomask = self.generate_mask(im, gt_mask, save_plot=False) # TODO make saveplot true sometimes
            # calculate metrics to evaluate model on test set
            generated_mask = torch.Tensor(pseudomask).int().view(400,400).to(self.device)
            groundtruth_mask = torch.Tensor(gt_mask).int().view(400,400).to(self.device)
            jaccard, accuracy, F1 = self.calc_metrics(generated_mask, groundtruth_mask)

            # unpack tuples of per class calculated metrics
            running_jaccard_nopalsa += jaccard[0]
            running_accuracy_nopalsa += accuracy[0]
            running_F1_nopalsa += F1[0]

            running_jaccard_palsa += jaccard[1]
            running_accuracy_palsa += accuracy[1]
            running_F1_palsa += F1[1]

        wandb.log({"test_jaccard_nopalsa": running_jaccard_nopalsa / len(test_loader.dataset)})
        wandb.log({"test_accuracy_nopalsa": running_accuracy_nopalsa / len(test_loader.dataset)})
        wandb.log({"test_F1_nopalsa": running_F1_nopalsa / len(test_loader.dataset)})

        wandb.log({"test_jaccard_palsa": running_jaccard_palsa / 107}) # hardcoded the num of samples in testdata that have palsa
        wandb.log({"test_accuracy_palsa": running_accuracy_palsa / 107}) # TODO maybe not make it hardcoded.. 
        wandb.log({"test_F1_palsa": running_F1_palsa / 107})

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
            input = arr[:,1,:,:].unsqueeze(1),
            scale_factor = im.shape[3]/arr.shape[3],
            mode='bilinear'
        ).cpu().detach()
        
        # select pixels based on activation threshold
        # if we include an image-specific threshold 
        if self.std_from_mean: 
            im_based_act_threshold = pals_acts.view(-1).mean() + self.std_from_mean * torch.std(pals_acts, dim=None)
            max_thresh = max([self.cam_threshold, im_based_act_threshold])
            pixels_activated = torch.where(torch.Tensor(pals_acts) > max_thresh, 1, 0).squeeze(0).permute(1,2,0).numpy()
         # if we use the global threshold 
        else:
            pixels_activated = torch.where(torch.Tensor(pals_acts) > self.cam_threshold, 1, 0).squeeze(0).permute(1,2,0).numpy()

        # Plot image with CAM
        cpu_img = im.squeeze().cpu().detach().permute(1,2,0).long().numpy()
        superpixels = np.array(snic(cpu_img, int(self.snic_seeds), self.snic_compactness)[0])
        pseudomask = self.create_superpixel_mask(superpixels, pixels_activated.squeeze(), threshold=self.overlap_threshold)

        if save_plot: self.generate_plot(cpu_img, pals_acts, im, pixels_activated, superpixels, pseudomask, gt)
        
        return pseudomask

    def generate_plot(self, cpu_img, pals_acts, im, pixels_activated, superpixels, pseudomask, gt):

        bounds = [0, 0.5, 1]
        overlap_bounds = [0, 0.5, 4, 11]

        cmap_cam = mcolors.ListedColormap(['lightblue', 'purple'])
        norm_cam = mcolors.BoundaryNorm(bounds, cmap_cam.N)

        cmap_snic = mcolors.ListedColormap(['lightblue', 'darkblue'])
        norm_snic = mcolors.BoundaryNorm(bounds, cmap_snic.N)

        cmap_overlap = mcolors.ListedColormap(['lightblue', 'darkblue', 'purple'])
        norm_overlap = mcolors.BoundaryNorm(overlap_bounds, cmap_overlap.N)

        # Create figure and gridspec
        fig = plt.figure(figsize=(51, 14))  # Adjust figure size as needed
        gs = fig.add_gridspec(2, 7)

        # RGB input
        ax0 = fig.add_subplot(gs[:, 0])
        ax0.imshow(cpu_img[:,:,:3])
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_title(f'RGB input', size = 22)

        # Hillshade input
        ax1 = fig.add_subplot(gs[:, 1])
        ax1.imshow(cpu_img[:,:,3], cmap = 'Greys')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f'Hillshade input', size = 22)

        # CAM
        ax2a = fig.add_subplot(gs[0, 2])
        ax2a.imshow(cpu_img[:,:,:3])
        ax2a.imshow(pals_acts.view(im.shape[3], im.shape[3], 1), alpha=.4, cmap='jet')
        ax2a.set_xticks([])
        ax2a.set_yticks([])
        ax2a.set_title(f'CAM', size = 22)

        # Activation mask
        ax2b = fig.add_subplot(gs[0, 3])
        ax2b.imshow(pixels_activated, cmap=cmap_cam, norm=norm_cam)
        ax2b.set_xticks([])
        ax2b.set_yticks([])
        ax2b.set_title(f'Activated cells', size = 22)

        # Superpixels
        boundaries = mark_boundaries(cpu_img[:,:,:3], superpixels)
        binary_boundaries = np.where(boundaries[:,:,0]>0.5, 1, 0)
        combined_image = np.copy(cpu_img[:,:,:3])
        combined_image[binary_boundaries == 1] = [0,0,139]

        # Superpixels on RGB
        ax3a = fig.add_subplot(gs[1, 2])
        ax3a.imshow(combined_image)
        ax3a.set_xticks([])
        ax3a.set_yticks([])
        ax3a.set_title(f'SNIC output', size = 22)

        # Superpixel borders
        ax3b = fig.add_subplot(gs[1, 3])
        ax3b.imshow(binary_boundaries, cmap=cmap_snic, norm=norm_snic)
        ax3b.set_xticks([])
        ax3b.set_yticks([])
        ax3b.set_title(f'Superpixels', size = 22)

        # Overlap
        overlap = np.copy(binary_boundaries)
        overlap[np.squeeze(pixels_activated) == 1] = 10
        ax4 = fig.add_subplot(gs[:, 4])
        ax4.imshow(overlap, cmap=cmap_overlap, norm=norm_overlap)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title(f'Overlap', size = 22)

        # Pseudomask
        ax5 = fig.add_subplot(gs[:, 5])
        ax5.imshow(pseudomask, cmap=cmap_cam, norm=norm_cam)
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.set_title(f'Pseudomask', size = 22)

        # Ground truth
        ax6 = fig.add_subplot(gs[:, 6])
        ax6.imshow(gt.squeeze(0).permute(1,2,0).long().numpy(), cmap=cmap_cam, norm=norm_cam)
        ax6.set_xticks([])
        ax6.set_yticks([])
        ax6.set_title(f'Ground truth', size = 22)


        plt.tight_layout()
        plt.show()

        # TODO CHANGE BACK TO BELOW BLOCK
        # plt.tight_layout()
        # wandb.log({'pseudomask': fig})
        # plt.close()


    def calc_metrics(self, pseudomask, gt):
        # Jaccard index (aka Intersection over Union - IoU) is the most common semantic seg metric
        jaccard = MulticlassJaccardIndex(num_classes=2, average=None).to(self.device)
        accuracy = MulticlassAccuracy(num_classes=2, average=None).to(self.device)
        F1 = MulticlassF1Score(num_classes=2, average=None).to(self.device)

        return jaccard(pseudomask, gt), accuracy(pseudomask, gt), F1(pseudomask, gt)

    def create_superpixel_mask(self, superpixels, binary_mask, threshold):
        # Get the unique superpixel labels
        unique_labels = np.unique(superpixels)

        # Create a dictionary to store the overlap percentage for each superpixel
        overlap_dict = {}

        # Iterate over each superpixel
        for label in unique_labels:
            # Create a mask for the current superpixel
            superpixel_mask = (superpixels == label)

            # Calculate the overlap percentage between activated and superpixel
            superpixel_size = np.sum(superpixel_mask)
            overlap_count = np.sum(superpixel_mask & binary_mask)
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
