import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json

import argparse
from models import DebugNN
from data import transform
import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(config):
    # Download the training and test datasets
    train_dataset = datasets.MNIST(root='datasets', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='datasets', train=False, download=True, transform=transform)

    train_dataset = Subset(train_dataset, list(range(1000)))

    # Create DataLoader for training and test sets
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = DebugNN()
    criterion = nn.BCEWithLogitsLoss()  # Combines a Sigmoid layer and the BCELoss
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["num_epochs"]):
        running_loss = 0.0
        model.train()
        for images, labels in tqdm.tqdm(train_loader):
            images = images.to(DEVICE)
            labels = (labels == 1)
            labels = labels.to(DEVICE).float().unsqueeze(1)  # Shape: (batch_size, 1)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Train loss: {running_loss/len(train_dataset):.4f}')

        model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(DEVICE)
                labels = (labels == 1)
                labels = labels.to(DEVICE).float().unsqueeze(1)  # Shape: (batch_size, 1)


                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Val loss: {val_running_loss/len(test_dataset):.3f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train / test mode", default="train")
    parser.add_argument("--config", type=str, help="Path to the training config")

    # Parse arguments
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = json.load(file)

    train(config)


