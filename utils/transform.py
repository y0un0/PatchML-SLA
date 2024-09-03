import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random
import matplotlib.pyplot as plt

class MultiResolutionPatches:
    def __init__(self, patch_size=64, stride=64, num_resolutions=3, downsample_ratio=2, max_patches_per_res=None, interpolation="bilinear"):
        self.patch_size = patch_size
        self.stride = stride
        self.num_resolutions = num_resolutions
        self.downsample_ratio = downsample_ratio
        self.max_patches_per_res = max_patches_per_res
        self.interpolation = interpolation
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)

    def extract_patches(self, image):
        patches = []
        # Go through each resolution level
        for r in range(self.num_resolutions):
            scale_factor = 1 / (self.downsample_ratio ** r)
            downsampled_image = F.interpolate(image.unsqueeze(0), scale_factor=scale_factor, mode=self.interpolation).squeeze()
            # Extract patches at resolution level
            patches_at_dim = self.unfold(downsampled_image)
            # Reshape
            channel, _, _ = downsampled_image.size()
            patches_at_dim = patches_at_dim.view(channel, self.patch_size, self.patch_size, -1)
            patches_at_dim = patches_at_dim.permute(3, 0, 1, 2)

            # Choose a number of patches to sample if the image size is too big
            if self.max_patches_per_res and self.max_patches_per_res > patches_at_dim.size(0):
                rand_indices = random.sample(range(patches_at_dim.size(0)), self.max_patches_per_res)
                patches_at_dim = patches_at_dim[rand_indices]

            patches.append(patches_at_dim)
        patches = torch.cat(patches)
        return patches

    def __call__(self, image):
        return self.extract_patches(image)

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    from torch.utils.data import Dataset, DataLoader

    patch_size=64
    stride=64
    num_resolutions=3
    downsample_ratio=2
    max_patches_per_res=None
    interpolation="bilinear"

    class CustomImageDataset(Dataset):
        def __init__(self, images, transform=None):
            self.images = images  # List of PIL images
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            image = transforms.ToTensor()(image)  # Convert PIL to Tensor
            if self.transform:
                patches = self.transform(image)
            return patches
        
    # Initialize patch extractor
    patch_extractor = MultiResolutionPatches(patch_size, stride, num_resolutions, downsample_ratio)

    # Example dataset and dataloader
    dummy_images = [Image.open(r"").resize((500, 500))]  # Create random dummy images
    dataset = CustomImageDataset(dummy_images, transform=patch_extractor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for patches in dataloader:
        print(patches.size())