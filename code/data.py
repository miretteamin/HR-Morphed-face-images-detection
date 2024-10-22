from torchvision import transforms
import torch
from torch.utils.data import Dataset
import os
from PIL import Image

from torchvision import transforms

# Define the transformation pipeline
trainval_transform = transforms.Compose([
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor and scale pixel values to [0.0, 1.0]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the tensor with mean and std
                         std=[0.229, 0.224, 0.225])
])

def read_paths_txt(file_path):
    image_paths = []
    labels = []

    with open(file_path, 'r') as file:
        for line in file:
            path, label = line.strip().split(' ')
            image_paths.append(path)
            labels.append(int(label))

    return image_paths, labels


class MorphDataset(Dataset):
    def __init__(self, dataset_dir, txt_paths, transform=trainval_transform):
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

