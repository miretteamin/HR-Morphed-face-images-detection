import os
import argparse
import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision import models
from data import MorphDataset, get_transforms
from metrics import MACER, BPCER, MACER_at_BPCER
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import numpy as np
from models import S2DCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path, model_name="efficientnet_b0"):
    """
    Load the model and weights from the checkpoint for the specified model architecture.
    """
    print(f"Loading model '{model_name}' from checkpoint: {checkpoint_path}")

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
    elif model_name == "mobilenetv3s":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 1)
    elif model_name == "s2d":
        model = S2DCNN()
    else:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. Please add it to the script."
        )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE)["model_state_dict"],
        strict=True,
    )
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
        "macer_at_bpcer": macer_at_bpcer,
    }


def main(args):
    with open(args.config, "r") as file:
        config = json.load(file)

    results_file = "results/evaluation_section.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {"models": []}

    print(f"Using device: {DEVICE}")
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Batch size: {args.batch_size}\n")

    weighted_metrics = {}
    total_samples = 0

    concat_datasets = [
        "E01_Global_val_lab.txt",
        "E04_Local_Match_val_lab.txt",
        "E05_DST_val_lab.txt",
        "E06_Twente_val_lab.txt",
        "E07_Lincoln_val_lab.txt",
        "E09_UNIBO_v2_val_lab.txt",
        "ManualMorphs_01_val_lab.txt",
    ]
    normal_datasets = ["FEI_val_lab.txt", "facelab_london_val_lab.txt"]
    basebio_dataset = "BaseBio_All_eval_lab.txt"

    model_entry = next(
        (m for m in all_results["models"] if m["model_name"] == args.model_name), None
    )

    if not model_entry:
        model_entry = {"model_name": args.name, "checkpoint": []}
        all_results["models"].append(model_entry)

    checkpoint_entry = next(
        (
            c
            for c in model_entry["checkpoint"]
            if c["model_checkpoint"] == os.path.basename(args.checkpoint)
        ),
        None,
    )

    if not checkpoint_entry:
        checkpoint_entry = {
            "model_checkpoint": os.path.basename(args.checkpoint),
            "val_datasets": [],
        }
        model_entry["checkpoint"].append(checkpoint_entry)

    for subset_file in config["val_txt"]:
        if subset_file == basebio_dataset:
            continue

        print(f"\nProcessing validation dataset: {subset_file}")

        if subset_file in concat_datasets:
            val_dataset = MorphDataset(
                dataset_dir=args.datadir,
                txt_paths=[subset_file, basebio_dataset],
                transform=get_transforms(is_train=False),
            )
        elif subset_file in normal_datasets:
            val_dataset = MorphDataset(
                dataset_dir=args.datadir,
                txt_paths=[subset_file],
                transform=get_transforms(is_train=False),
            )
        else:
            print(f"Unknown dataset: {subset_file}")
            continue

        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        num_samples = len(val_dataset)
        print(f"Loaded validation dataset with {num_samples} samples.")

        model = load_model(args.checkpoint, model_name=args.model_name)
        labels, outputs = evaluate(model, val_loader)

        metrics = compute_metrics(labels, outputs)

        checkpoint_entry["val_datasets"].append(
            {
                "dataset_name": subset_file,
                "metrics": metrics,
                "num_samples": num_samples,
            }
        )

        for key, value in metrics.items():
            weighted_metrics[key] = weighted_metrics.get(key, 0) + value * num_samples
        total_samples += num_samples

    final_metrics = {
        key: value / total_samples for key, value in weighted_metrics.items()
    }

    print("\nWeighted Metrics Across All Validation Datasets:")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the training config file."
    )
    parser.add_argument(
        "--datadir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="efficientnet_b0",
        help="Model architecture (default: efficientnet_b0).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation (default: 128).",
    )
    parser.add_argument(
        "--name", type=str, default="efficientnet_b0", help="Model name in the json."
    )

    args = parser.parse_args()

    main(args)
