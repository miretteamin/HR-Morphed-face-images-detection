import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.models as models

import json
import argparse
import tqdm


from data import MorphDataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def train(config):
    train_dataset = MorphDataset(dataset_dir=config["dataset_dir"], txt_paths=config["train_txt"])
    val_dataset = MorphDataset(dataset_dir=config["dataset_dir"], txt_paths=config["val_txt"])

    print(f"Total number of train objects: {len(train_dataset)}.")
    print(f"Total number of val objects: {len(val_dataset)}.")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    print(model)
    print(f"Total number of parameters in the model: {count_parameters(model)}.")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["pos_weight"]))  # Combines a Sigmoid layer and the BCELoss
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["num_epochs"]):
        running_loss = 0.0
        model.train()
        for batch_idx, (images, labels) in enumerate(tqdm.tqdm(train_loader)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Train loss: {running_loss/len(train_dataset):.10f}')

        model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            for batch_idx, (images, labels) in enumerate(tqdm.tqdm(val_loader)):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())

                val_running_loss += loss.item()

            torch.save({
                'epoch': epoch + 1,  # Save the epoch number
                'model_state_dict': model.state_dict(),  # Save model state
                'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state (optional)
                'val_loss': val_running_loss / len(val_dataset),  # Save current validation loss
            }, os.path.join(config["save_dir"], f"model_epoch_{epoch+1}.pth"))

        print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Val loss: {val_running_loss/len(val_dataset):.3f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train / test mode", default="train")
    parser.add_argument("--config", type=str, help="Path to the training config")
    parser.add_argument("--datadir", type=str, help="Path to the training config")

    # Parse arguments
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = json.load(file)

    os.mkdir(f"./logs/{config['name']}")

    config["dataset_dir"] = args.datadir
    config["save_dir"] = f"./logs/{config['name']}"
    train(config)


