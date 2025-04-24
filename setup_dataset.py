import os
from datasets import load_dataset

# Define the correct path to the extracted dataset
data_dir = "/home/codespace/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray"

# Check the dataset structure
if os.path.exists(data_dir):
    print(f"Dataset directory exists: {data_dir}")
    print("Subdirectories:", os.listdir(data_dir))  # Check subfolders like 'train', 'test', etc.
else:
    print(f"Dataset directory not found at {data_dir}")

# Load dataset from the correct directory
dataset = load_dataset("imagefolder", data_dir=data_dir)

# Print out the dataset to verify it's loading correctly
print(dataset)
