from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((64, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Adjust normalization as per dataset
])

