import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from xml.dom import minidom

import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import cv2

class PascalVOC(Dataset):

    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        'tvmonitor', 'ambiguous'
    ]

    CLASS_IDX = { 
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3,
            'bottle': 4,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'chair': 8,
            'cow': 9,
            'diningtable': 10,
            'dog': 11,
            'horse': 12,
            'motorbike': 13,
            'person': 14,
            'pottedplant': 15,
            'sheep': 16,
            'sofa': 17,
            'train': 18,
            'tvmonitor': 19
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

    def __init__(self, cfg, split, test_mode=False, resize_type="default", root=os.path.expanduser('./data')):
        super(VOCSingleAnnot, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.test_mode = test_mode
        self.resize_type = resize_type

        # train/val/test splits are pre-cut
        if self.split == 'train':
            _split_f = os.path.join(self.root, 'ImageSets', 'Main', 'train.txt')
        elif self.split == 'val' or self.split == 'test':
            _split_f = os.path.join(self.root, 'ImageSets', 'Main', 'val.txt')
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
            
            
                _one_hot_label = os.path.join(self.root, "Annotations",  _one_hot_label)
                assert os.path.isfile(_one_hot_label), '%s not found' % _one_hot_label
                self.one_hot_labels.append(_one_hot_label)

        assert (len(self.images) == len(self.one_hot_labels))
        if self.split == 'train':
            assert len(self.images) == 5717
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(0.4,0.4,0.4,0.2, p=0.2),
                A.Normalize(PascalVOC.MEAN, PascalVOC.STD),
                ToTensorV2()
            ])
        elif self.split == 'val' or self.split == "test":
            assert len(self.images) == 5823
            self.transform = A.Compose([
                A.Normalize(PascalVOC.MEAN, PascalVOC.STD),
                ToTensorV2()
            ])

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """
            Resize input image and keep its aspect ratio. The image szie need to be divisible by 32 (stride=32).
        """
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle --> False allows to keep a squared input
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

    
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, top, bottom, left, right

    def read_xml(self, one_hot_label_file):
        doc = minidom.parse(one_hot_label_file)
        items = doc.getElementsByTagName("object")
        ground_truths = [item.getElementsByTagName("name")[0].firstChild.nodeValue for item in items]
        ground_truths = [PascalVOC.CLASS_IDX[name] for name in ground_truths]
        return ground_truths


    def one_hot_encoding(self, multiclass_labels):
        """
        Read the xml file find all the label in the image and 
        select the unique label by performing a random uniform selection
        """
        # Remove the duplicate
        multiclass_labels = np.unique(multiclass_labels)
        if len(multiclass_labels) > 1 and 14 in multiclass_labels:
            multiclass_labels = np.delete(multiclass_labels, np.argwhere(multiclass_labels == 14))
        if self.test_mode:
            # In test mode, no need to randomly choose one label
            kept_class = multiclass_labels
        else:
            kept_class = [random.choice(multiclass_labels)]
        one_hot_label = np.zeros(PascalVOC.NUM_CLASS)
        for cls_idx in range(PascalVOC.NUM_CLASS):
            if cls_idx in kept_class:
                one_hot_label[cls_idx] = 1
        return one_hot_label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = Image.open(self.images[index]).convert('RGB')
        image = np.array(image)
        if self.resize_type == "letterbox":
            image, _, _, _, _ = self.letterbox(image, new_shape=(640, 640), auto=False)
        else:
            image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)
        image = self.transform(image=image)["image"]
        cls_info = self.read_xml(self.one_hot_labels[index])
        one_hot_label = self.one_hot_encoding(cls_info)
        return image, one_hot_label

    @property
    def pred_offset(self):
        return 0

if __name__ == "__main__":
    from xml.dom import minidom

    dataset = VOCSingleAnnot(cfg=None, split="val", test_mode=None, root="/teamspace/studios/this_studio/PatchML-SLA/data/VOC2012/")
    print(dataset[100][1])