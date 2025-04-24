import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Define paths
zip_path = "/tmp/chest-xray-pneumonia.zip"
extract_path = "/tmp/chest_xray"

def download_and_extract():
    # Check if already extracted
    if os.path.exists(os.path.join(extract_path, "chest_xray")):
        print("✅ Dataset already extracted.")
        return

    # Authenticate and download using Kaggle API
    print("🔑 Authenticating Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(zip_path):
        print("⬇️  Downloading dataset...")
        api.dataset_download_files(
            'paultimothymooney/chest-xray-pneumonia',
            path='/tmp',
            unzip=False
        )
        print("✅ Download complete.")
    else:
        print("📦 Dataset already downloaded.")

    # Extract ZIP
    print("🗂️  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Extraction complete.")

if __name__ == "__main__":
    download_and_extract()