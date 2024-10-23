import torch.nn as nn
import torch.nn.functional as F


class DebugNN(nn.Module):
    def __init__(self):
        super(DebugNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1,
                               padding=1)  # Output: 16 x 64 x 96
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,
                               padding=1)  # Output: 32 x 32 x 48
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                               padding=1)  # Output: 64 x 16 x 24
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