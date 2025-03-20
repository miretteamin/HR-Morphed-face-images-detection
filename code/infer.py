import argparse
import torch
import tqdm
from torchvision import models
from data import get_transforms
import os

from metrics import MACER, BPCER, MACER_at_BPCER
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import numpy as np

from PIL import Image

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


def read_paths_txt(file_path):
    image_paths = []
    labels = []

    with open(file_path, "r") as file:
        for line in file:
            path, label = line.strip().split(" ")
            image_paths.append(path)
            labels.append(int(label))

    return image_paths, labels


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
        "macer_at_bpcer@0.01": macer_at_bpcer,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "macer": macer,
        "bpcer": bpcer,
    }


def infer(args):
    transform = get_transforms(is_train=False)
    model = load_model(args.checkpoint, args.model_name)
    model.eval()
    # print(f"Loaded model {args.model_name} from checkpoint {args.checkpoint}.")

    image_paths, labels = read_paths_txt(args.datafile)
    print(f"In file {args.datafile} found {len(image_paths)} images.")

    print("Starting infering the images...")

    result = {}
    result["metrics"] = None
    result["predictions"] = {}

    all_labels, all_preds = [], []
    for i in tqdm.tqdm(range(len(image_paths))):
        image_path, label = image_paths[i], labels[i]

        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)

        pred = torch.sigmoid(model(image)).squeeze().item()

        all_labels.append(label)
        all_preds.append(pred)
        result["predictions"][image_path] = (label, pred)

    result["metrics"] = compute_metrics(all_labels, all_preds)

    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, args.save)

    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datafile",
        type=str,
        required=True,
        help="Path to the .txt file with images paths.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model architechture of the saved checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--save", type=str, default="result.json", help="Path of the save .json file"
    )

    args = parser.parse_args()

    infer(args)
