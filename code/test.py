import torch
import wandb
import os
from models import DebugNN
from data import MorphDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import json
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Transformation pipeline
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Accessing wandb
wandb.login(key='6b2c400d545f1a22b720d687700b682f3f433b55')

def load_wandb_checkpoint(run_id, epoch):
    """
    Load a model checkpoint from wandb given a specific run ID and epoch.
    
    Parameters:
        run_id (str): The wandb run ID.
        epoch (int): The specific epoch to load the checkpoint for.
    
    Returns:
        model (torch.nn.Module): The model loaded with the checkpoint weights.
    """
    # Initialize wandb run object
    run = wandb.init(id=run_id, project="face-morph-detection", entity="mirettemoawad-ecole-polytechnique", resume="allow")
    
    # Download the checkpoint file from wandb
    checkpoint_path = run.use_artifact(f"model_epoch_{epoch}.pth:latest").download()
    
    # Define your model
    model = DebugNN()  # Replace with your model if different
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, f"model_epoch_{epoch}.pth"))['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    
    wandb.finish()
    
    return model

# Test function to evaluate model checkpoints
def test_model(config_path, run_id, epochs):
    """
    Test the model across specified epochs and print performance metrics.
    
    Parameters:
        config_path (str): Path to the configuration file.
        run_id (str): wandb run ID for fetching checkpoints.
        epochs (list[int]): List of epochs to test.
    """
    # Load config file
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    # Load test dataset
    test_dataset = MorphDataset(dataset_dir=config["dataset_dir"], txt_paths=config["val_txt"], transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    
    # Iterate through each specified epoch
    for epoch in epochs:
        print(f"Testing model at epoch {epoch}")
        
        # Load model checkpoint for the current epoch
        model = load_wandb_checkpoint(run_id, epoch)
        
        # Evaluation metrics storage
        total_loss = 0
        all_labels = []
        all_outputs = []
        
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["pos_weight"]).to(device=DEVICE))
        
        # Evaluation loop
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item()
            
            # Collect labels and predictions for metrics
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.squeeze().cpu().numpy())
                
        print(f"Epoch {epoch} Test Loss: {total_loss / len(test_loader)}")
        
        # Optionally, visualize or log metrics to wandb if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file with model and dataset settings")
    parser.add_argument("--datadir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, nargs='+', required=True, help="List of epochs to test (e.g., 1 5 10)")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = json.load(file)

    os.makedirs(f"./logs/{config['name']}", exist_ok=True)

    config["dataset_dir"] = args.datadir
    config["save_dir"] = f"./logs/{config['name']}"

    test_model(config, args.run_id, args.epochs)

