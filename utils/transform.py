import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange

class MultiResolutionPatches:
    def __init__(self, patch_size=64, stride=64, num_resolutions=3, downsample_ratio=2, max_patches_per_res=None, interpolation="bilinear"):
        self.patch_size = patch_size
        self.stride = stride
        self.num_resolutions = num_resolutions
        self.downsample_ratio = downsample_ratio
        self.max_patches_per_res = max_patches_per_res
        self.interpolation = interpolation
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)

    def extract_patches(self, images):
        """
        Extract patches from a batch of images at multiple resolutions.
        @param images: Tensor of shape [batch_size, channels, height, width]
        @return: Patches of shape [batch_size * num_patches, channels, patch_height, patch_width]
        """
        patches = []
        batch_size, channels, h, w = images.size()
        
        # Go through each resolution level
        for r in range(self.num_resolutions):
            scale_factor = 1 / (self.downsample_ratio ** r)
            # Downsample the entire batch of images
            downsampled_images = F.interpolate(images, scale_factor=scale_factor, mode=self.interpolation, align_corners=False)
            # Extract patches at the current resolution
            patches_at_res = self.unfold(downsampled_images)
            # Reshape patches
            num_patches = patches_at_res.size(-1)

            patches_at_res = patches_at_res.view(batch_size, channels, self.patch_size, self.patch_size, num_patches)
            patches_at_res = patches_at_res.permute(0, 4, 1, 2, 3)
            # Sample a subset of patches if needed
            if self.max_patches_per_res and num_patches > self.max_patches_per_res:
                selected_indices = torch.randperm(num_patches)[:self.max_patches_per_res]
                patches_at_res = patches_at_res[selected_indices]
            
            patches.append(patches_at_res)
        
        # Concatenate patches from all resolutions
        patches = torch.cat(patches, dim=1)
        batch_size, num_patches_tot, channels, ph, pw = patches.shape
        patches = patches.reshape(batch_size * num_patches_tot, channels, ph, pw)
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