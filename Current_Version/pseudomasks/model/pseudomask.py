import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
import wandb
from pysnic.algorithms.snic import snic
from skimage.segmentation import mark_boundaries
from torch.autograd import Variable
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics import Accuracy, F1Score

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

        running_jaccard = 0
        running_accuracy = 0
        running_F1 = 0

        for im, lab, _, gt_mask in test_loader:
            pseudomask = self.generate_mask(im, gt_mask, save_plot=False) # TODO make saveplot true sometimes
            # calculate metrics to evaluate model on test set
            generated_mask = torch.Tensor(pseudomask).int().view(400,400).to(self.device)
            groundtruth_mask = torch.Tensor(gt_mask).int().view(400,400).to(self.device)
            jaccard, accuracy, F1 = self.calc_metrics(generated_mask, groundtruth_mask)
            running_jaccard += jaccard
            running_accuracy += accuracy
            running_F1 += F1

        wandb.log({"test_mean_jaccard": running_jaccard / len(test_loader.dataset)})
        wandb.log({"test_mean_accuracy": running_accuracy / len(test_loader.dataset)})
        wandb.log({"test_mean_F1": running_F1 / len(test_loader.dataset)})

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

        print(f"maximum activation = {torch.max(pals_acts)}") #TODO remove this line

        # Plot image with CAM
        cpu_img = im.squeeze().cpu().detach().permute(1,2,0).long().numpy()
        superpixels = np.array(snic(cpu_img, int(self.snic_seeds), self.snic_compactness)[0])
        pseudomask = self.create_superpixel_mask(superpixels, pixels_activated.squeeze(), threshold=self.overlap_threshold)

        if save_plot: self.generate_plot(cpu_img, pals_acts, im, pixels_activated, superpixels, pseudomask, gt)
        
        return pseudomask

    def generate_plot(self, cpu_img, pals_acts, im, pixels_activated, superpixels, pseudomask, gt):

        # Plotting fucntion to visualize the pseudomask generation process (and intermediates)
        cmap = mcolors.ListedColormap(['black', 'lightblue'])
        bounds = [0, 0.5, 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

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

        ax3.imshow(pixels_activated, cmap=cmap, norm=norm)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title('Activated cells')

        ax4.imshow(mark_boundaries(cpu_img[:,:,:3], superpixels)) # TODO change so image is actually plotted.
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title('Superpixels')

        ax5.imshow(pseudomask, cmap=cmap, norm=norm)
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.set_title('Pseudomask')

        ax6.imshow(gt.squeeze(0).permute(1,2,0).long().numpy(), cmap=cmap, norm=norm)
        ax6.set_xticks([])
        ax6.set_yticks([])
        ax6.set_title('Ground Truth')

        plt.tight_layout()
        plt.show()

        # TODO: restore original block below (remove above)
        # plt.tight_layout()
        # wandb.log({'pseudomask': fig})
        # plt.close()

    def calc_metrics(self, pseudomask, gt):
        # Jaccard index (aka Intersection over Union - IoU) is the most common semantic seg metric
        jaccard = MulticlassJaccardIndex(num_classes=2).to(self.device)
        accuracy = Accuracy(task="multiclass", num_classes=2).to(self.device)
        F1 = F1Score(task="multiclass", num_classes=2).to(self.device)

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
