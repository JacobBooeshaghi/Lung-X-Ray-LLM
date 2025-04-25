from dataloaders import get_dataloaders

print("Testing dataloader setup...")
try:
    # Try to get the dataloaders with a small batch size
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
    
    # Print dataset sizes
    print(f"\nDataset sizes:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Try to get one batch to verify data format
    print("\nTesting batch loading...")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print("\nSetup successful! The dataloaders are working correctly.")
except Exception as e:
    print(f"Error: {e}") 