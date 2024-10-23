import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import json
import argparse
import tqdm
import wandb


from data import MorphDataset
from metrics import F1_Score, MACER, BPCER
from models import DebugNN


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.login(key='b720adf497c3c34f4e18be46e08aaba5ff31321b')


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def train(config):

    wandb.init(project="face-morph-detection", config=config, entity="mirettemoawad-ecole-polytechnique", name=config["name"])

    print("Device: ", DEVICE)

        
    train_dataset = MorphDataset(dataset_dir=config["dataset_dir"], txt_paths=config["train_txt"])
    val_dataset = MorphDataset(dataset_dir=config["dataset_dir"], txt_paths=config["val_txt"])

    train_dataset = Subset(train_dataset, random.sample(range(0, len(train_dataset)), 50000))
    val_dataset =  Subset(val_dataset, random.sample(range(0, len(val_dataset)), 50000))

    print(f"Total number of train objects: {len(train_dataset)}.")
    print(f"Total number of val objects: {len(val_dataset)}.")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    # model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)

    model = DebugNN()
    print(model)
    model = model.to(DEVICE)

    print(f"Total number of parameters in the model: {count_parameters(model)}.")
    model.eval()

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["pos_weight"]).to(device=DEVICE))  # Combines a Sigmoid layer and the BCELoss
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

            if batch_idx % 100 == 0:
                # Calculate metrics
                all_labels = labels.cpu().numpy()
                all_outputs = outputs.squeeze().detach().cpu().numpy()

                preds = (torch.sigmoid(outputs) >= 0.5).int().cpu().numpy() 

                acc = accuracy_score(all_labels, preds)
                prec = precision_score(all_labels, preds)
                rec = recall_score(all_labels, preds)

                # f1 = F1_Score(all_labels, all_outputs)
                f1 = f1_score(all_labels, all_outputs > 0.5)

                macer = MACER(all_labels, all_outputs)
                bpcer = BPCER(all_labels, all_outputs)

                print("F1 Score: ", f1, "----- MACER: ", macer, "---- BPCER: ", bpcer, "---- Accuracy: ", acc, "---- Precision: ", prec, "---- Recall: ", rec)

                # Log metrics with wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": running_loss / (batch_idx+1),
                    "f1_score": f1,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "macer": macer,
                    "bpcer": bpcer
                })
            
        train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Train loss: {running_loss/len(train_dataset):.10f}')

        model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            all_labels = []
            all_outputs = []

            for batch_idx, (images, labels) in enumerate(tqdm.tqdm(val_loader)):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())

                val_running_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.squeeze().cpu().numpy())

            torch.save({
                'epoch': epoch + 1,  # Save the epoch number
                'model_state_dict': model.state_dict(),  # Save model state
                'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state (optional)
                'val_loss': val_running_loss / len(val_dataset),  # Save current validation loss
            }, os.path.join(config["save_dir"], f"model_epoch_{epoch+1}.pth"))
        
        val_loss = val_running_loss / len(val_loader)
        print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Val loss: {val_running_loss/len(val_dataset):.3f}')
    

        # Calculate metrics
        all_labels = np.array(all_labels)
        all_outputs = np.array(all_outputs)

        f1 = F1_Score(all_labels, all_outputs)
        macer = MACER(all_labels, all_outputs)
        bpcer = BPCER(all_labels, all_outputs)

        # Log metrics with wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy_val": acc,
            "precision_val": prec,
            "recall_val": rec,
            "f1_score_val": f1,
            "macer_val": macer,
            "bpcer_val": bpcer
        })

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train / test mode", default="train")
    parser.add_argument("--config", type=str, help="Path to the training config")
    parser.add_argument("--datadir", type=str, help="Path to the training config")

    # Parse arguments
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = json.load(file)

    # os.mkdir(f"./logs/{config['name']}")
    os.makedirs(f"./logs/{config['name']}", exist_ok=True)

    config["dataset_dir"] = args.datadir
    config["save_dir"] = f"./logs/{config['name']}"
    train(config)


