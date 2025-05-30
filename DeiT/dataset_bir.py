import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.vision import VisionDataset

###########################################################

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    # classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    # if not classes:
    #     raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    # class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    class_idx_file = os.path.join(directory, "class_idx.txt")
    class_to_idx = {}
    classes = []
    with open(class_idx_file, "r") as file:
        for line in file:
            k = line.split(":")[0]
            v = int(line.split(":")[1].strip("\n"))
            class_to_idx[k] = v
            classes.append(k)
    return classes, class_to_idx

## raw
# def make_dataset(
#     directory: str,
#     split: str,
#     subset: str,
#     anno_split: int,
#     channel: int,
#     class_to_idx: Optional[Dict[str, int]] = None,
#     extensions: Optional[Union[str, Tuple[str, ...]]] = None,
#     is_valid_file: Optional[Callable[[str], bool]] = None,
# ) -> List[Tuple[str, int]]:
#     """Generates a list of samples of a form (path_to_sample, class).

#     See :class:`DatasetFolder` for details.

#     Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
#     by default.
#     """
#     directory = os.path.expanduser(directory)

#     if class_to_idx is None:
#         _, class_to_idx = find_classes(directory)
#     elif not class_to_idx:
#         raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

#     both_none = extensions is None and is_valid_file is None
#     both_something = extensions is not None and is_valid_file is not None
#     if both_none or both_something:
#         raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

#     if extensions is not None:

#         def is_valid_file(x: str) -> bool:
#             return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

#     is_valid_file = cast(Callable[[str], bool], is_valid_file)

#     # instances = []
#     # available_classes = set()
#     # for target_class in sorted(class_to_idx.keys()):
#     #     class_index = class_to_idx[target_class]
#     #     target_dir = os.path.join(directory, target_class)
#     #     if not os.path.isdir(target_dir):
#     #         continue
#     #     for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
#     #         for fname in sorted(fnames):
#     #             path = os.path.join(root, fname)
#     #             if is_valid_file(path):
#     #                 item = path, class_index
#     #                 instances.append(item)

#     #                 if target_class not in available_classes:
#     #                     available_classes.add(target_class)

#     # empty_classes = set(class_to_idx.keys()) - available_classes
#     # if empty_classes:
#     #     msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
#     #     if extensions is not None:
#     #         msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
#     #     raise FileNotFoundError(msg)

#     instances = []
#     instances_file_path = os.path.join(directory, "annotations-{}/split-{}/{}.txt".format(subset, anno_split, split))
#     with open(instances_file_path, "r") as file:
#         for line in file:
#             line_arr = line.strip("\n").split()
#             img_name = line_arr[0]
#             vessel_dir = "{}_FLAC".format(img_name[0:4])
#             channel_dir = "CHANNEL_{}".format(channel)
#             label_id = int(line_arr[1])
#             if img_name.endswith("_1.jpg"):
#                 img_name = img_name.replace("_1.jpg", "_{}.jpg".format(channel))
#             img_path = os.path.join(directory, vessel_dir, channel_dir, img_name)
#             item = img_path, label_id
#             instances.append(item)


    

#     return instances

def make_dataset(
    directory: str,
    split: str,
    subset: str,
    anno_split: int,
    channel: int,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    instances_file_path = os.path.join(directory, "annotation-{}/split-{}/{}.txt".format(subset, anno_split, split))
    print(instances_file_path)
    with open(instances_file_path, "r") as file:
        for line in file:
            line_arr = line.strip("\n").split()
            img_name = line_arr[0]  # 提取图像名称，包括路径
            label_id = int(line_arr[1])  # 提取类别标签
            img_path = os.path.join(directory, img_name)  # 拼接根目录和图像名称，生成完整路径
            item = img_path, label_id  # 创建样本条目
            instances.append(item)  # 添加样本条目到列表

    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        split: str,
        subset: str,
        anno_split: str,
        channel: int,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes("{}/annotation-{}/split-{}".format(self.root, subset, anno_split))
        samples = self.make_dataset(self.root, split, subset, anno_split, channel, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        split: str,
        subset: str,
        anno_split: str,
        channel: int,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, split, subset, anno_split, channel, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageAnnoFolder(DatasetFolder):
    """A generic data loader where the image meta files are arranged in this way by default: ::

        root/annotation/train.txt
        root/annotation/val.txt

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        split: str,
        subset: str,
        anno_split: str,
        channel: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            split,
            subset,
            anno_split,
            channel,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
