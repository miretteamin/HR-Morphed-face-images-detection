import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from models import DebugNN, S2DCNN

def get_model(model_name):
    """Facade to create a model for the training

    Args:
        model_name (str): should be from a list of available models

    Returns:
        torch.nn: model
    """
    if model_name == "debugnn":
        model = DebugNN()

    elif model_name == "s2dcnn":
        model = S2DCNN()

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)

    elif model_name == "resnet18":
        model = models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)

    elif model_name == "resnet34":
        model = models.resnet34(weights="DEFAULT")
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)

    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 1)

    return model


def get_optimizer(config, model):
    if config["optimizer"] == "adam":
        return optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    elif config["optimizer"] == "sgd":
        return optim.SGD(model.parameters(), lr=config["learning_rate"])    
    

def get_scheduler(config, optimizer):
    lr_scheduler = None
    
    if config["scheduler"] == "CosineAnnealingLR":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=20)
        
    elif config["scheduler"] == "ReduceLROnPlateau":
        lr_scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
    
    elif config["scheduler"] == "StepLR":
        lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)    
        
    return lr_scheduler