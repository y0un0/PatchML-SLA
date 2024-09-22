import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImagePalette

class PascalVOC(Dataset):

    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambiguous'
    ]

    CLASS_IDX = {
            'background': 0, 
            'aeroplane': 1,
            'bicycle': 2,
            'bird': 3,
            'boat': 4,
            'bottle': 5,
            'bus': 6,
            'car': 7,
            'cat': 8,
            'chair': 9,
            'cow': 10,
            'diningtable': 11,
            'dog': 12,
            'horse': 13,
            'motorbike': 14,
            'person': 15,
            'potted-plant': 16,
            'sheep': 17,
            'sofa': 18,
            'train': 19,
            'tv/monitor': 20
            }

    CLASS_IDX_INV = {v: k for k, v in CLASS_IDX.items()}

    NUM_CLASS = 20

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()

    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image


class VOCSingleAnnot(PascalVOC):

    def __init__(self, cfg, split, test_mode, root=os.path.expanduser('./data')):
        super(VOCSingleAnnot, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.test_mode = test_mode

        # train/val/test splits are pre-cut
        if self.split == 'train':
            _split_f = os.path.join(self.root, 'ImageSets', 'Segmentation', 'train.txt')
        elif self.split == 'val':
            _split_f = os.path.join(self.root, 'ImageSets', 'Segmentation', 'val.txt')
        elif self.split == 'test':
            _split_f = os.path.join(self.root, 'ImageSets', 'Segmentation', 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        assert os.path.isfile(_split_f), "%s not found" % _split_f

        self.images = []
        self.one_hot_labels = []
        with open(_split_f, "r") as lines:
            for line in lines:
                _image = line.strip("\n").split(' ')[0] + ".jpg"
                _one_hot_label = line.strip("\n").split(' ')[0] + ".xml"
                _image = os.path.join(self.root, "JPEGImages", _image)
                assert os.path.isfile(_image), '%s not found' % _image
                self.images.append(_image)
            
                if self.split != 'test':
                    _one_hot_label = os.path.join(self.root, "Annotations",  _one_hot_label)
                    assert os.path.isfile(_one_hot_label), '%s not found' % _one_hot_label
                    self.one_hot_labels.append(_one_hot_label)

        if self.split != 'test':
            assert (len(self.images) == len(self.one_hot_labels))
            if self.split == 'train':
                assert len(self.images) == 10582
            elif self.split == 'val':
                assert len(self.images) == 1449

        # self.transform = tf.Compose([tf.MaskRandResizedCrop(self.cfg.DATASET), \
        #                              tf.MaskHFlip(), \
        #                              tf.MaskColourJitter(p = 1.0), \
        #                              tf.MaskNormalise(self.MEAN, self.STD), \
        #                              tf.MaskToTensor()])
    def one_hot_encoding(self):
        """
        Read the xml file find all the label in the image and 
        select the unique label by performing a random uniform selection
        """
        pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = Image.open(self.images[index]).convert('RGB')
        # TODO: process the xml file
        one_hot_label  = self.one_hot_labels[index]

        # general resize, normalize and toTensor
        # image, one_hot_label = self.transform(image, one_hot_label)

        return image, one_hot_label, os.path.basename(self.images[index])

    @property
    def pred_offset(self):
        return 0

if __name__ == "__main__":
    dataset = VOCSingleAnnot(cfg=None, split="val", test_mode=None, root="/teamspace/studios/this_studio/PatchML-SLA/data/VOC2012/")
    print(dataset[0])