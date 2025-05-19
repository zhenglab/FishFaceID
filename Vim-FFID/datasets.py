# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from torchvision.datasets.folder import ImageFolder

class CustomDataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader):
        # 为训练集和测试集设置不同的路径
        subfolder = 'train' if train else 'test'
        super().__init__(os.path.join(root, subfolder), transform=transform, target_transform=target_transform, loader=loader)
        print(f"加载数据集目录: {os.path.join(root, subfolder)}")
        print(f"类别映射: {self.class_to_idx}")
        print(f"类别顺序: {self.classes}")

    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        # 自定义排序规则，确保fish_1, fish_2, ..., fish_10的正确顺序
        classes.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else x)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class CUB200Dataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader):
        # CUB200-2011 数据集的 train/test 分割信息
        subfolder = 'train' if train else 'test'
        path_images = os.path.join(root, 'images')
        path_splits = os.path.join(root, 'train_test_split.txt')
        image_labels = os.path.join(root, 'image_class_labels.txt')

        # 读取图片标签和训练/测试分割信息
        with open(image_labels, 'r') as f:
            labels = {line.split()[0]: int(line.split()[1]) - 1 for line in f.readlines()}

        with open(path_splits, 'r') as f:
            splits = {line.split()[0]: int(line.split()[1]) for line in f.readlines()}

        # 筛选出符合当前模式（训练或测试）的图片
        selected_imgs = [
            img for img, is_train in splits.items() if (train and is_train == 1) or (not train and is_train == 0)
        ]

        self.samples = [
            (os.path.join(path_images, f"{img}.jpg"), labels[img]) for img in selected_imgs if img in labels
        ]
        self.targets = [s[1] for s in self.samples]  # 兼容某些 PyTorch 函数

        super().__init__(root=path_images, transform=transform, target_transform=target_transform, loader=loader)

        self.imgs = self.samples  # ImageFolder 需要使用 imgs 属性

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform,download=True)
        nb_classes = 100
    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform,download=True)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
    
    elif args.data_set == 'custom':
        dataset = CustomDataset(args.data_path, train=is_train, transform=transform)
        nb_classes = len(dataset.classes)
    
    elif args.data_set == 'CUB200':
        dataset = CUB200Dataset(args.data_path, train=is_train, transform=transform)
        nb_classes = 200  # CUB200-2011 has 200 classes
    

   




    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)






# # Copyright (c) 2015-present, Facebook, Inc.
# # All rights reserved.





# import os
# import json

# from torchvision import datasets, transforms
# from torchvision.datasets.folder import ImageFolder, default_loader

# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.data import create_transform


# class INatDataset(ImageFolder):
#     def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
#                  category='name', loader=default_loader):
#         self.transform = transform
#         self.loader = loader
#         self.target_transform = target_transform
#         self.year = year
#         # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
#         path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
#         with open(path_json) as json_file:
#             data = json.load(json_file)

#         with open(os.path.join(root, 'categories.json')) as json_file:
#             data_catg = json.load(json_file)

#         path_json_for_targeter = os.path.join(root, f"train{year}.json")

#         with open(path_json_for_targeter) as json_file:
#             data_for_targeter = json.load(json_file)

#         targeter = {}
#         indexer = 0
#         for elem in data_for_targeter['annotations']:
#             king = []
#             king.append(data_catg[int(elem['category_id'])][category])
#             if king[0] not in targeter.keys():
#                 targeter[king[0]] = indexer
#                 indexer += 1
#         self.nb_classes = len(targeter)

#         self.samples = []
#         for elem in data['images']:
#             cut = elem['file_name'].split('/')
#             target_current = int(cut[2])
#             path_current = os.path.join(root, cut[0], cut[2], cut[3])

#             categors = data_catg[target_current]
#             target_current_true = targeter[categors[category]]
#             self.samples.append((path_current, target_current_true))

#     # __getitem__ and __len__ inherited from ImageFolder


# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)

#     if args.data_set == 'CIFAR':
#         dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
#         nb_classes = 100
#     elif args.data_set == 'IMNET':
#         root = os.path.join(args.data_path, 'train' if is_train else 'val')
#         dataset = datasets.ImageFolder(root, transform=transform)
#         nb_classes = 1000
#     elif args.data_set == 'custom':
#         root = os.path.join(args.data_path, 'train' if is_train else 'test')
#         dataset = datasets.ImageFolder(root, transform=transform)
#         nb_classes = 5
#     elif args.data_set == 'INAT':
#         dataset = INatDataset(args.data_path, train=is_train, year=2018,
#                               category=args.inat_category, transform=transform)
#         nb_classes = dataset.nb_classes
#     elif args.data_set == 'INAT19':
#         dataset = INatDataset(args.data_path, train=is_train, year=2019,
#                               category=args.inat_category, transform=transform)
#         nb_classes = dataset.nb_classes

#     return dataset, nb_classes


# def build_transform(is_train, args):
#     resize_im = args.input_size > 32
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation=args.train_interpolation,
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#         )
#         if not resize_im:
#             # replace RandomResizedCropAndInterpolation with
#             # RandomCrop
#             transform.transforms[0] = transforms.RandomCrop(
#                 args.input_size, padding=4)
#         return transform

#     t = []
#     if resize_im:
#         size = int(args.input_size / args.eval_crop_ratio)
#         t.append(
#             transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
#         )
#         t.append(transforms.CenterCrop(args.input_size))

#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
#     return transforms.Compose(t)

