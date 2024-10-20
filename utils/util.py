import math
import os

from torch import nn
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import MultiResolutionPatches

class PatchDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images  # List of PIL images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            patches = self.transform(image)
        return patches

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
    if steps > 0:
        lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_dataloader(args, trainset, validset):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    return trainloader, validloader

def get_patches(args, inputs):
    # Initialize patch extractor
    patch_extractor = MultiResolutionPatches(args.patch_size, args.stride, args.num_resolution, 
                                             args.downsample_ratio, interpolation=args.interpolation)
    # Generate the set of patches
    patches_set = PatchDataset(inputs, transform=patch_extractor)
    patches_loader = torch.utils.data.DataLoader(patches_set, batch_size=args.batch_size, shuffle=False)
    for patches in patches_loader:
        return patches