import torch
from torchvision import datasets, transforms
import os

def get_dataloaders(batch_size=16):
    # Correct path to the extracted dataset
    dataset_dir = "/home/codespace/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray"

    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets from the correct directories
    train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
