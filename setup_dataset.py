import os
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil

def setup_dataset():
    # Define the path where we want to store the dataset
    data_dir = os.path.join(os.path.dirname(__file__), "data", "chest_xray")
    
    # Check if dataset already exists
    if os.path.exists(data_dir) and os.path.exists(os.path.join(data_dir, "train")) and \
       os.path.exists(os.path.join(data_dir, "test")) and os.path.exists(os.path.join(data_dir, "val")):
        print(f"Dataset already exists at {data_dir}")
        return data_dir
    
    print("Downloading dataset from Kaggle...")
    
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    try:
        # Download the dataset
        api.dataset_download_files(
            'paultimothymooney/chest-xray-pneumonia',
            path='data',
            unzip=True
        )
        print("Dataset downloaded successfully!")
        
        # The dataset should now be in data/chest_xray
        if os.path.exists(data_dir):
            print(f"Dataset extracted to: {data_dir}")
            print("Available directories:", os.listdir(data_dir))
        else:
            raise Exception(f"Dataset not found at expected location: {data_dir}")
            
        return data_dir
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

if __name__ == "__main__":
    # Run the setup when script is run directly
    setup_dataset()
