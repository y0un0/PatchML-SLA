import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class SyntheticDataset:
    def __init__(self, image_shape, n_cls, n_samples, split, seed=42, max_overlap=0.1):
        self.image_shape = image_shape
        self.n_cls = n_cls
        self.n_samples = n_samples
        self.seed = seed # For reproducibility
        self.max_overlap = max_overlap
        self.split = split

        self.images, self.one_hot_labels = self.generate_samples()

        if self.split == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        elif self.split == 'val':
            self.transform = A.Compose([
                A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ToTensorV2()
            ])

    def generate_colormap(self):
        np.random.seed(self.seed)  # For reproducibility
        colormap = np.random.randint(50, 256, size=(self.n_cls, 3))
        return colormap 

    def generate_samples(self):
        assert len(self.image_shape) == 3 and self.image_shape[2] == 3, "Image shape must be (H, W, 3)."
        
        height, width, _ = self.image_shape
        colormap = self.generate_colormap()

        images = []
        labels = []
        for n in range(self.n_samples):
            image = np.zeros(self.image_shape, dtype=np.uint8)
            label = np.zeros(self.n_cls, dtype=np.int32)
            
            occupied_mask = np.zeros((height, width), dtype=np.bool_)
            
            # Randomly add blocks of class colors to the image
            num_blocks = np.random.randint(2, self.n_cls + 1)
            for _ in range(num_blocks):
                class_idx = np.random.randint(0, self.n_cls)
                color = colormap[class_idx]
                
                # Creating the blocks for each classes and checking if they overlap too much
                for _ in range(50):
                    block_height = np.random.randint(height // 4, height // 2)
                    block_width = np.random.randint(width // 4, width // 2)
                    start_x = np.random.randint(0, height - block_height)
                    start_y = np.random.randint(0, width - block_width)
                    
                    # Computing the overlap of the different blocks
                    block_mask = np.zeros((height, width), dtype=np.bool_)
                    block_mask[start_x:start_x + block_height, start_y:start_y + block_width] = True
                    overlap = np.logical_and(occupied_mask, block_mask).sum()
                    overlap_ratio = overlap / (block_height * block_width)
                    
                    if overlap_ratio <= self.max_overlap:
                        # Adding the block in the image
                        image[start_x:start_x + block_height, start_y:start_y + block_width] = color
                        occupied_mask[start_x:start_x + block_height, start_y:start_y + block_width] = True
                        label[class_idx] = 1
                        break
            
            images.append(image)
            labels.append(label)
        return images, labels
    
    def pseudo_one_hot_label(self, one_hot_label):
        non_zeros_idx = np.argwhere(one_hot_label == 1)
        pseudo_one_hot_label = np.zeros(self.n_cls, dtype=np.int32)
        kept_cls = random.choice(non_zeros_idx)
        pseudo_one_hot_label[kept_cls] = 1
        return pseudo_one_hot_label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = self.transform(image=image)["image"]
        one_hot_label = self.one_hot_labels[index]
        pseudo_one_hot_label = self.pseudo_one_hot_label(one_hot_label)
        pseudo_one_hot_label = np.array(pseudo_one_hot_label)
        return image, pseudo_one_hot_label
        

if __name__ == "__main__":
    dataset = SyntheticDataset(image_shape=(640, 640, 3), n_cls=3, n_samples=200, split='train', seed=42, max_overlap=0.1)
    print(dataset[0])
