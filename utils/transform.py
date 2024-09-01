import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
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
            downsampled_image = F.interpolate(image.unsqueeze, scale_factor=scale_factor, mode=self.interpolation)
            # Extract patches at resolution level
            patches_at_dim = self.unfold(downsampled_image)
            # Reshape
            batch_size, channel, _, _ = downsampled_image.size()
            patches_at_dim = patches_at_dim.view(batch_size, channel, self.patch_size, self.patch_size, -1)
            patches_at_dim = patches_at_dim.permute(0, 4, 1, 2, 3)

            # TODO: Handle max number of patches
            pass

    def __call__(self):
        pass

if __name__ == "__main__":
    patch_size=64
    stride=64
    num_resolutions=3
    downsample_ratio=2
    max_patches_per_res=None
    interpolation="bilinear"

    patches = []
    image_size = (500, 500)
    from PIL import Image
    image = Image.open(r"")
    image = image.resize((500, 500))
    image = transforms.ToTensor()(image)
    for i in range(num_resolutions):
        scale_factor = 1 / (downsample_ratio ** i)
        new_image = F.interpolate(image.unsqueeze(0), scale_factor=scale_factor, mode="bilinear")
        print("nn.functional new_size: ", new_image.size())
        unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)
        patches_at_dim = unfold(new_image)
        patches_at_dim = patches_at_dim.view(patches_at_dim.size(0), 
                                             3, 
                                             patch_size, 
                                             patch_size, 
                                             -1)
        patches_at_dim = patches_at_dim.permute(0, 4, 1, 2, 3)
        print(patches_at_dim.size())