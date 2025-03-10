import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import json
import argparse
import tqdm
import wandb


from data import get_transforms, MorphDataset, MorphDatasetMemmap
from metrics import MACER, BPCER, MACER_at_BPCER
from train_utils import get_model, get_optimizer, get_scheduler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def init_wandb(config, run_name):
    if not config["activate"]:
        return

    if "wandb_key" in config:
        wandb.login(key=config["wandb_key"])

    if config["wandb_run_id"]:
        wandb.init(
            project=config["project"],
            config=config,
            entity=config["entity"],
            id=config["wandb_run_id"],
            resume="allow",
        )

    else:
        wandb.init(
            project=config["project"],
            config=config,
            entity=config["entity"],
            name=run_name,
        )


def train(config):
    init_wandb(config["wandb"], config["name"])

    print("#### Device ####: ", DEVICE)

    trains_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)

    if config["is_memmap"]:
        train_dataset = MorphDatasetMemmap(
            config["train_memmap"], config["train_labels"], transform=trains_transform
        )
        val_dataset = MorphDatasetMemmap(
            config["val_memmap"], config["val_labels"], transform=val_transform
        )

    else:
        train_dataset = MorphDataset(
            dataset_dir=config["dataset_dir"],
            txt_paths=config["train_txt"],
            transform=trains_transform,
        )
        val_dataset = MorphDataset(
            dataset_dir=config["dataset_dir"],
            txt_paths=config["val_txt"],
            transform=val_transform,
        )

    print(f"Total number of train objects: {len(train_dataset)}.")
    print(f"Total number of val objects: {len(val_dataset)}.")

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    model = get_model(config["model_name"])

    if config["weights_path"]:
        checkpoint = torch.load(config["weights_path"])
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model weights from {config['weights_path']}.")
        start_epoch = config["begin_epoch"]
    else:
        start_epoch = 0

    print(f"Total number of parameters in the model: {count_parameters(model)}.")

    model = model.to(DEVICE)
    model.eval()

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config["pos_weight"]).to(device=DEVICE)
    )
    optimizer = get_optimizer(config, model)
    lr_scheduler = get_scheduler(config, optimizer)

    print("Starting epoch: ", start_epoch)

    for epoch in range(start_epoch, config["num_epochs"]):
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
                all_labels = labels.cpu().numpy()
                all_outputs = outputs.squeeze().detach().cpu().numpy()

                preds = (torch.sigmoid(outputs) >= 0.5).int().cpu().numpy()

                acc = accuracy_score(all_labels, preds)
                prec = precision_score(all_labels, preds, zero_division=0)
                rec = recall_score(all_labels, preds, zero_division=0)

                f1 = f1_score(all_labels, all_outputs > 0.5, zero_division=0)

                macer = MACER(all_labels, all_outputs)
                bpcer = BPCER(all_labels, all_outputs)

                macer_at_bpcer = MACER_at_BPCER(all_labels, all_outputs)

                print(
                    "F1 Score: ",
                    f1,
                    "----- MACER: ",
                    macer,
                    "---- BPCER: ",
                    bpcer,
                    "--- MACER@BPCER=1%: ",
                    macer_at_bpcer,
                    "---- Accuracy: ",
                    acc,
                    "---- Precision: ",
                    prec,
                    "---- Recall: ",
                    rec,
                )
                if config["wandb"]["activate"]:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train_loss": running_loss / (batch_idx + 1),
                            "f1_score": f1,
                            "accuracy": acc,
                            "precision": prec,
                            "recall": rec,
                            "macer": macer,
                            "bpcer": bpcer,
                            "macer_at_bpcer": macer_at_bpcer,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )

        train_loss = running_loss / len(train_loader)
        print(
            f"Epoch [{epoch + 1}/{config['num_epochs']}], Train loss: {running_loss / len(train_dataset):.10f}"
        )

        if config["scheduler"] == "ReduceLROnPlateau":
            lr_scheduler.step(train_loss)
        else:
            lr_scheduler.step()

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

            val_loss = val_running_loss / len(val_loader)
            print(
                f"Epoch [{epoch + 1}/{config['num_epochs']}], Val loss: {val_running_loss / len(val_dataset):.3f}"
            )

            all_labels = np.array(all_labels)
            all_outputs = np.array(all_outputs)

            preds = (
                (torch.sigmoid(torch.tensor(all_outputs, dtype=torch.float)) >= 0.5)
                .int()
                .cpu()
                .numpy()
            )

            acc_val = accuracy_score(all_labels, preds)
            prec_val = precision_score(all_labels, preds, zero_division=0)
            rec_val = recall_score(all_labels, preds, zero_division=0)

            f1_val = f1_score(all_labels, preds, zero_division=0)

            macer_val = MACER(all_labels, all_outputs)
            bpcer_val = BPCER(all_labels, all_outputs)

            macer_at_bpcer_val = MACER_at_BPCER(all_labels, all_outputs)

            print(
                "F1 Score Val: ",
                f1_val,
                "----- MACER Val: ",
                macer_val,
                "---- BPCER Val: ",
                bpcer_val,
                "--- MACER@BPCER=1%_val: ",
                macer_at_bpcer_val,
                "---- Accuracy Val: ",
                acc_val,
                "---- Precision Val: ",
                prec_val,
                "---- Recall Val: ",
                rec_val,
            )

            checkpoint_path = os.path.join(
                config["save_dir"], f"model_epoch_{epoch + 1}.pth"
            )
            try:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": running_loss / len(train_loader),
                    },
                    checkpoint_path,
                )
                print(f"✅ Model checkpoint saved: {checkpoint_path}")
            except Exception as e:
                print(f"❌ Error saving model checkpoint at {checkpoint_path}: {e}")
            if config["wandb"]["activate"]:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss_val": train_loss,
                        "val_loss": val_loss,
                        "accuracy_val": acc_val,
                        "precision_val": prec_val,
                        "recall_val": rec_val,
                        "f1_score_val": f1_val,
                        "macer_val": macer_val,
                        "bpcer_val": bpcer_val,
                        "macer_at_bpcer_val": macer_at_bpcer_val,
                    }
                )
    if config["wandb"]["activate"]:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train / test mode", default="train")
    parser.add_argument("--memmap", help="memmap enable", action="store_true")
    parser.add_argument("--config", type=str, help="Path to the training config")

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = json.load(file)

    config["save_dir"] = os.path.join(config["save_dir"], config["name"])
    config["is_memmap"] = args.memmap
    config["weights_path"] = args.weights
    config["wandb_run_id"] = args.wandb_run_id

    os.makedirs(config["save_dir"], exist_ok=True)

    train(config)
