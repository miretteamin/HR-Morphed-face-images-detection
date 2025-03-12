from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


IMAGE_SHAPE = (64, 96, 3)


def read_paths_txt(file_path):
    image_paths = []
    labels = []

    with open(file_path, "r") as file:
        for line in file:
            path, label = line.strip().split(" ")
            image_paths.append(path)
            labels.append(int(label))

    return image_paths, labels


class DownscaleBilinear:
    def __call__(self, img):
        width, height = img.size
        new_width, new_height = width // 2, height // 2
        return img.resize((new_width, new_height))


def get_transforms(is_train=True):
    """Function to generate the augmentations pipeline for the train or val modes

    Args:
        is_train (bool, optional): set True if train mode. Defaults to True.

    Returns:
        transforms.Compose: transforms.Compose pipeline
    """
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.3),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class MorphDataset(Dataset):
    """
    Dataset to operate with images located in a data directory
    """

    def __init__(self, dataset_dir, txt_paths, transform=None):
        """

        Args:
            dataset_dir (str): path to the dataset directory
            txt_paths (list): list of the images sub-directories (set up in config)
            transform (optional): augmentation pipeline for the images. Defaults to None.
        """
        self.dataset_dir = dataset_dir
        self.txt_paths = txt_paths
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self._collect_data()

    def _collect_data(self):
        for txt_path in self.txt_paths:
            new_paths, new_labels = read_paths_txt(
                os.path.join(self.dataset_dir, txt_path)
            )
            self.image_paths.extend(new_paths)
            self.labels.extend(new_labels)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Generates one sample of data."""

        img_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(os.path.join(self.dataset_dir, img_path)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class MorphDatasetMemmap(Dataset):
    """
    This dataset class is used to operate with numpy memory map file as a dataset (np.memmap)
    """

    def __init__(self, memmap_path, labels_path, transform=None, is_train=True):
        """

        Args:
            memmap_path: path to the .dat file with the dataset (np memory map)
            labels_path: path to the .npy file that contains labels for the corresponding images from np.memmap file
            transform (optional): optional augmentations
            is_train (optional): train or val mode
        """
        self.transform = (
            transform if transform is not None else get_transforms(is_train)
        )
        self.labels = np.load(labels_path)
        self.data = np.memmap(
            memmap_path,
            dtype=np.uint8,
            mode="r",
            shape=(len(self.labels), *IMAGE_SHAPE),
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = np.array(self.data[idx])
        image = Image.fromarray(image)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
