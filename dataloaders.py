import torch
from torchvision import datasets, transforms
import os
from setup_dataset import setup_dataset
import random
from torch.utils.data import Subset

def get_balanced_indices(dataset, samples_per_class=1000):
    # Get all indices for each class
    class_indices = {i: [] for i in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Print class distribution before balancing
    print("\nOriginal class distribution:")
    for class_idx in class_indices:
        print(f"Class {dataset.classes[class_idx]}: {len(class_indices[class_idx])} samples")
    
    # Randomly sample equal number from each class
    balanced_indices = []
    for class_idx in class_indices:
        indices = class_indices[class_idx]
        if len(indices) > samples_per_class:
            indices = random.sample(indices, samples_per_class)
        balanced_indices.extend(indices)
    
    # Print final distribution
    print("\nBalanced class distribution:")
    balanced_count = {i: 0 for i in range(len(dataset.classes))}
    for idx in balanced_indices:
        _, label = dataset[idx]
        balanced_count[label] += 1
    for class_idx in balanced_count:
        print(f"Class {dataset.classes[class_idx]}: {balanced_count[class_idx]} samples")
    
    return balanced_indices

def get_dataloaders(batch_size=16, augment=False, samples_per_class=1000):
    # Set random seed for reproducibility
    random.seed(42)
    
    # Get dataset directory from setup_dataset
    dataset_dir = setup_dataset()

    # Define transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Validation and test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load full datasets with appropriate transforms
    full_train_dataset = datasets.ImageFolder(
        os.path.join(dataset_dir, 'train'),
        transform=train_transform
    )
    
    # Get balanced indices for training
    balanced_indices = get_balanced_indices(full_train_dataset, samples_per_class)
    
    # Create balanced training dataset
    train_dataset = Subset(full_train_dataset, balanced_indices)
    
    # Load validation and test datasets normally
    val_dataset = datasets.ImageFolder(
        os.path.join(dataset_dir, 'val'),
        transform=val_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(dataset_dir, 'test'),
        transform=val_transform
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
