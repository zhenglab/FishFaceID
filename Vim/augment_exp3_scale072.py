# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
3Augment implementation
Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino)
and timm DA(https://github.com/rwightman/pytorch-image-models)
"""
import torch
from torchvision import transforms

# error: cannot import name '_pil_interp' from 'timm.data.transforms' 
# from timm.data.transforms import _pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor

# fix: timm version problem
# from timm.data.transforms import str_pil_interp as _pil_interp
from timm.data.transforms import RandomResizedCropAndInterpolation, ToNumpy, ToTensor

import numpy as np
from torchvision import datasets, transforms
import random

from datasets_mean_std import SEACUM_MEAN_DICT, SEACUM_STD_DICT, GROUPER_MEAN_DICT, GROUPER_STD_DICT, SEACUCUMBER_MEAN_DICT, SEACUCUMBER_STD_DICT, CORALGROUPER_MEAN_DICT, CORALGROUPER_STD_DICT, BLUEGROUPER_MEAN_DICT, BLUEGROUPER_STD_DICT

from PIL import ImageFilter, ImageOps
import torchvision.transforms.functional as TF


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
 
    
    
class horizontal_flip(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2,activate_pred=False):
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
        
    
    
def new_data_aug_generator(args = None):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("你正在运行new_data_aug_generator函数")
    img_size = args.input_size
    remove_random_resized_crop = args.src
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    subset = args.subset
    print(subset)
    if args.species=='SeaCum':
        mean = SEACUM_MEAN_DICT[subset]
        std = SEACUM_STD_DICT[subset]
    elif args.species=='Grouper':
        mean = GROUPER_MEAN_DICT[subset]
        std = GROUPER_STD_DICT[subset]
    elif args.species=='SeaCucumber':
        mean = SEACUCUMBER_MEAN_DICT[subset]
        std = SEACUCUMBER_STD_DICT[subset]
    elif args.species=='CoralGrouper':
        mean = CORALGROUPER_MEAN_DICT[subset]
        std = CORALGROUPER_STD_DICT[subset]
    elif args.species=='BlueGrouper':
        mean = BLUEGROUPER_MEAN_DICT[subset]
        std = BLUEGROUPER_STD_DICT[subset]
    print(mean)
    print(std)
    primary_tfl = []
    scale=(0.08, 0.72)
    interpolation='bicubic'
    if remove_random_resized_crop:
        primary_tfl = [
            transforms.Resize(img_size, interpolation=3),
            transforms.RandomCrop(img_size, padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ]
    else:
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, interpolation=interpolation),
            transforms.RandomHorizontalFlip()
        ]

        
    secondary_tfl = [transforms.RandomChoice([gray_scale(p=1.0),
                                              Solarization(p=1.0),
                                              GaussianBlur(p=1.0)])]
   
    if args.color_jitter is not None and not args.color_jitter==0:
        secondary_tfl.append(transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter))
    final_tfl = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
    return transforms.Compose(primary_tfl+secondary_tfl+final_tfl)
