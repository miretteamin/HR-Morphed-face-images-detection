from torchvision import transforms
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

from torchvision import transforms

IMAGE_SHAPE = (64, 96, 3)

# # Define the transformation pipeline
# trainval_transform = transforms.Compose([
#     transforms.ToTensor(),          # Convert the image to a PyTorch tensor and scale pixel values to [0.0, 1.0]
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the tensor with mean and std
#                          std=[0.229, 0.224, 0.225])
# ])

def read_paths_txt(file_path):
    image_paths = []
    labels = []

    with open(file_path, 'r') as file:
        for line in file:
            path, label = line.strip().split(' ')
            image_paths.append(path)
            labels.append(int(label))

    return image_paths, labels


class DownscaleBilinear:
    def __call__(self, img):
        width, height = img.size
        new_width, new_height = width // 2, height // 2
        return img.resize((new_width, new_height), resample=Image.BILINEAR)


def get_transforms(is_train = True):
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p = 0.3),
            transforms.RandomRotation(degrees = 10, p = 0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            DownscaleBilinear(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    else:
        return transforms.Compose([
            DownscaleBilinear(),                         
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.5], std=[0.5])])
            ])


class MorphDataset(Dataset):
    def __init__(self, dataset_dir, txt_paths, transform=None):
        self.dataset_dir = dataset_dir
        self.txt_paths = txt_paths
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self._collect_data()

    def _collect_data(self):
        for txt_path in self.txt_paths:
            new_paths, new_labels = read_paths_txt(os.path.join(self.dataset_dir, txt_path))
            self.image_paths.extend(new_paths)
            self.labels.extend(new_labels)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Generates one sample of data."""
        # Get the image path and label for the given index
        img_path = self.image_paths[index]
        label = self.labels[index]

        # Load the image using PIL
        image = Image.open(os.path.join(self.dataset_dir, img_path)).convert('RGB')  # Convert image to RGB if it's not

        # Apply any transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label


class MorphDatasetMemmap(Dataset):
    def __init__(self, memmap_path, labels_path, transform=trainval_transform):
        self.transform = transform
        self.labels = np.load(labels_path)
        self.data = np.memmap(memmap_path, dtype=np.uint8, mode='r', shape=(len(self.labels), *IMAGE_SHAPE))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = np.array(self.data[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
