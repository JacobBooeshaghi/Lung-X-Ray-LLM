import os
import random
import shutil

def balance_directory(src_dir, max_images=1000):
    """Balance a directory by keeping only max_images random images."""
    # List all images
    images = [f for f in os.listdir(src_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
    print(f"\nProcessing {src_dir}")
    print(f"Found {len(images)} images")
    
    if len(images) > max_images:
        # Randomly select images to keep
        keep_images = set(random.sample(images, max_images))
        # Remove excess images
        for img in images:
            if img not in keep_images:
                os.remove(os.path.join(src_dir, img))
                print(f"Removed: {img}")
    
    print(f"Final count: {len([f for f in os.listdir(src_dir) if f.endswith(('.jpeg', '.jpg', '.png'))])}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Paths
    data_dir = "data/chest_xray"
    train_normal = os.path.join(data_dir, "train/NORMAL")
    train_pneumonia = os.path.join(data_dir, "train/PNEUMONIA")
    
    # Balance training directories
    print("Balancing training dataset...")
    balance_directory(train_normal, 1000)
    balance_directory(train_pneumonia, 1000)
    
    print("\nDataset balancing completed!")

if __name__ == "__main__":
    main() 