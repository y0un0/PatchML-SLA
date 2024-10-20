import argparse
import math
import os
import time
from datetime import datetime
import logging
import pprint

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import numpy as np
import timm.optim

from datasets import VOCSingleAnnot
from models import PatchMLSL
from utils import (WeakNegativeLoss, get_dataloader, 
                    AverageMeter, adjust_learning_rate, get_patches)

parser = argparse.ArgumentParser(description='Training PatchMLSL')
parser.add_argument('--device', default="cuda", type=str, help='which device to use')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--dataset', default="VOC2012", type=str, help='dataset name')
parser.add_argument('--dataset_path', default="/teamspace/studios/this_studio/PatchML-SLA/data/VOC2012/", type=str, help='location of dataset')
parser.add_argument('--model', default='efficientnet_b4', type=str, help='model architecture: [efficientnet_b0, ...]')
parser.add_argument('--n_blocks', default=5, type=int, help='number of blocks to keep in the architecture')
parser.add_argument('--intermediate_dim', default=128, type=int, help='intermediate dim')
parser.add_argument('--embed_dim', default=256, type=int, help='embedded dim')
parser.add_argument('--theta', default=0.0, type=float, help='threshold relu for the weak negative loss')
parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
parser.add_argument('--save-epoch', default=5, type=int, help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int, help='mini-batch size (default: 16)')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--optimizer', default="lamb", type=str, help="name of the optimizer to use")
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 0.0001)')
parser.add_argument('--patch-size', default=64, type=int, help='height and width of the patch')
parser.add_argument('--stride', default=64, type=int, help='number of pixels that separates each patch')
parser.add_argument('--num-resolution', default=3, type=int, help='number of times to downsample the image')
parser.add_argument('--downsample-ratio', default=2, type=int, help='uniform ratio applied for downsampling')
parser.add_argument('--interpolation', default='bilinear', type=str, help='interpolation used for downsampling')

# LR scheduler
parser.add_argument('--lr_decay_epochs', type=str, default='5,10,15,20', help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay rate for learning rate')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

date_time = datetime.now().strftime("%d_%m_%H:%M")

# Processing str to list for epochs to decay learning rate 
args.lr_decay_epochs = [int(step) for step in args.lr_decay_epochs.split(',')]

args.name = date_time + "_" + 'weaknegative_{}_lr_{}_linear_bsz_{}_{}_{}_{}_{}'.\
        format(args.model, args.learning_rate,
               args.batch_size, args.epochs, args.intermediate_dim, args.embed_dim, args.dataset)

args.log_directory = "logs/{in_dataset}/{name}/".format(in_dataset=args.dataset, name=args.name)
args.model_directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.dataset, name=args.name)

if not os.path.exists(args.model_directory):
    os.makedirs(args.model_directory)
if not os.path.exists(args.log_directory):
    os.makedirs(args.log_directory)

#save args
with open(os.path.join(args.log_directory, 'train_args.txt'), 'w') as f:
    f.write(pprint.pformat(state))

#init log
log = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s : %(message)s')
fileHandler = logging.FileHandler(os.path.join(args.log_directory, "train_info.log"), mode='w')
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
log.setLevel(logging.DEBUG)
log.addHandler(fileHandler)
log.addHandler(streamHandler) 

log.debug(state)

if args.dataset == "VOC2012":
    args.n_cls = 20
elif args.dataset == "COCO":
    args.n_cls = 80

#set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
log.debug(f"{args.name}")

def main():

    if args.dataset == "VOC2012":
        trainset = VOCSingleAnnot(cfg=None, split="train", test_mode=None, root=args.dataset_path)
        validset = VOCSingleAnnot(cfg=None, split="val", test_mode=None, root=args.dataset_path)
        trainloader, validloader = get_dataloader(args, trainset, validset)
    
    elif args.dataset == "COCO":
        pass
    
    model = PatchMLSL(model_name=args.model, n_blocks=args.n_blocks, intermediate_dim=args.intermediate_dim, 
                      embed_dim=args.embed_dim, n_cls=args.n_cls)
    model = model.to(args.device)

    criterion_wnl = WeakNegativeLoss(theta=args.theta)
    if args.optimizer == "lamb":
        optimizer = timm.optim.Lamb(params=model.parameters(), lr=args.learning_rate, 
                                weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, 
                                weight_decay=args.weight_decay)

    # train(args, trainloader, model)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        # TODO: Implement the training loop
        pass
    
def train(args, trainloader, model, criterion_wnl, optimizer, epoch, log):
# def train(args, trainloader, model):
    batch_time = AverageMeter()
    wnl_losses = AverageMeter()

    model.train()
    end = time.time()
    for i, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        # Dividing the input image into patches
        input_patches = get_patches(args, inputs)

        logits = model(input_patches)
        # TODO: Modify the forward function of the model to return the image embeddings (for the loss)
        pass

if __name__ == "__main__":
    main()