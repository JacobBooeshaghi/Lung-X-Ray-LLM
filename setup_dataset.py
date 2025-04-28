import os
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil
from tqdm import tqdm
import requests
from pathlib import Path
import zipfile
import time

def download_with_progress(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
        with open(filename, 'wb') as f:
            for data in response.iter_content(chunk_size=4096):
                f.write(data)
                pbar.update(len(data))

def setup_dataset():
    # Define the path where we want to store the dataset
    data_dir = os.path.join(os.path.dirname(__file__), "data", "chest_xray")
    
    # Check if dataset already exists
    if os.path.exists(data_dir) and os.path.exists(os.path.join(data_dir, "train")) and \
       os.path.exists(os.path.join(data_dir, "test")) and os.path.exists(os.path.join(data_dir, "val")):
        print(f"Dataset already exists at {data_dir}")
        return data_dir
    
    print("Setting up Kaggle API...")
    
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    temp_zip = os.path.join("data", "chest_xray.zip")
    
    try:
        print("\nStarting download... This may take a while as the dataset is large.")
        print("Download progress will be shown below:")
        
        # Download the dataset with progress tracking
        api.dataset_download_files(
            'paultimothymooney/chest-xray-pneumonia',
            path='data',
            unzip=False,  # We'll handle unzipping ourselves
            quiet=False
        )
        
        # Move the downloaded file to our temp location
        downloaded_file = os.path.join("data", "chest-xray-pneumonia.zip")
        if os.path.exists(downloaded_file):
            shutil.move(downloaded_file, temp_zip)
        
        print("\nExtracting dataset...")
        # Extract the zip file
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            # Get total number of files for progress bar
            total_files = len(zip_ref.namelist())
            with tqdm(total=total_files, desc="Extracting") as pbar:
                for file in zip_ref.namelist():
                    zip_ref.extract(file, "data")
                    pbar.update(1)
        
        # Clean up the zip file
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
        
        print("\nDataset downloaded and extracted successfully!")
        
        # The dataset should now be in data/chest_xray
        if os.path.exists(data_dir):
            print(f"Dataset extracted to: {data_dir}")
            print("Available directories:", os.listdir(data_dir))
        else:
            raise Exception(f"Dataset not found at expected location: {data_dir}")
            
        return data_dir
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Clean up any partial downloads
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
        raise

if __name__ == "__main__":
    # Run the setup when script is run directly
    setup_dataset()
