import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32, image_size=224):
    data_dir = "/tmp/chest_xray/chest_xray"  # this is where the extracted dataset should be

    # ViT expects 3 channels and standardized input size
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # convert 1-channel to 3
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset  = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader