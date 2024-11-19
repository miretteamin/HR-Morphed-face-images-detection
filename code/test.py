import os
import argparse
import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision import models
from data import MorphDataset, trainval_transform
from metrics import MACER, BPCER, MACER_at_BPCER
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path):
    """
    Load the model and weights from the checkpoint.
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE)['model_state_dict'])
    model = model.to(DEVICE)
    return model


def evaluate(model, val_loader):
    """
    Evaluate the model on the validation dataset.
    """
    model.eval()
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in tqdm.tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images).squeeze()
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())

    return all_labels, all_outputs


def compute_metrics(labels, outputs, threshold=0.5):
    """
    Compute all metrics based on predictions and ground truths.
    """
    outputs = np.array(outputs)
    labels = np.array(labels)

    preds = (outputs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    macer = MACER(labels, outputs, threshold)
    bpcer = BPCER(labels, outputs, threshold)
    macer_at_bpcer = MACER_at_BPCER(labels, outputs, target_bpcer=0.01)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "macer": macer,
        "bpcer": bpcer,
        "macer_at_bpcer": macer_at_bpcer
    }


def main(args):
    with open(args.config, 'r') as file:
        config = json.load(file)

    val_dataset = MorphDataset(
        dataset_dir=args.datadir,
        txt_paths=config["val_txt"],
        transform=trainval_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    print(f"Loaded validation dataset with {len(val_dataset)} samples.")

    model = load_model(args.checkpoint)

    labels, outputs = evaluate(model, val_loader)

    metrics = compute_metrics(labels, outputs)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the training config file.")
    parser.add_argument("--datadir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")

    args = parser.parse_args()

    main(args)
