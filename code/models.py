import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class DebugNN(nn.Module):
    def __init__(self):
        super(DebugNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )  # Output: 16 x 64 x 96
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # Output: 32 x 32 x 48
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # Output: 64 x 16 x 24
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 8 * 12, 128)
        self.fc2 = nn.Linear(128, 1)  # Output single logit for binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 16 x 32 x 48
        x = self.pool(F.relu(self.conv2(x)))  # 32 x 16 x 24
        x = self.pool(F.relu(self.conv3(x)))  # 64 x 8 x 12
        x = x.view(-1, 64 * 8 * 12)  # Flatten
        x = F.relu(self.fc1(x))  # 128
        x = self.fc2(x)  # 1
        return x




class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        B, C, H, W = x.size()
        block_size = self.block_size
        # Check if spatial dimensions are divisible by the block size

        assert H % block_size == 0 and W % block_size == 0, (
            "Height and Width must be divisible by block size"
        )

        # Rearrange the tensor
        x = x.view(B, C, H // block_size, block_size, W // block_size, block_size)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * (block_size**2), H // block_size, W // block_size)
        return x


class S2DCNN(nn.Module):
    def __init__(self):
        super(S2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )  # Output: 16 x 64 x 96
        self.space_to_depth1 = SpaceToDepth(
            block_size=2
        )  # Reduces spatial size, increases depth
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # Adjusted input channels
        self.space_to_depth2 = SpaceToDepth(block_size=2)
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # Adjusted input channels

        # Fully connected layers
        self.fc1 = nn.Linear(
            64 * 4 * 6 * 16, 128
        )  # Match flattened feature size after conv3
        self.fc2 = nn.Linear(128, 1)  # Single logit output for binary classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.space_to_depth1(x)
        x = F.relu(self.conv2(x))
        x = self.space_to_depth2(x)
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

