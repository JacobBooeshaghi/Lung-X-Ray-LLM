import os
import subprocess
import sys
import time
from pathlib import Path
import shutil

class XRayPipeline:
    def __init__(self):
        self.steps = [
            self.setup_kaggle_key,
            self.setup_dataset,
            self.train_model,
            self.launch_ui
        ]
        self.current_step = 0
        self.total_steps = len(self.steps)
        
    def setup_kaggle_key(self):
        """Step 1: Setup Kaggle API key"""
        print("\n=== Step 1/4: Setting up Kaggle API key ===")
        try:
            # Check if kaggle.json exists in the current directory
            local_key_path = Path("kaggle.json")
            if not local_key_path.exists():
                print("Error: kaggle.json not found in the current directory.")
                print("Please place your kaggle.json file in the project root directory.")
                return False

            # Create .kaggle directory in user's home directory if it doesn't exist
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(exist_ok=True)

            # Copy kaggle.json to the correct location
            target_key_path = kaggle_dir / "kaggle.json"
            shutil.copy2(local_key_path, target_key_path)

            # Set correct permissions (read/write for user only)
            target_key_path.chmod(0o600)

            print("Kaggle API key setup complete!")
            return True
        except Exception as e:
            print(f"Error setting up Kaggle API key: {str(e)}")
            return False

    def setup_dataset(self):
        """Step 2: Setup and download the dataset"""
        print("\n=== Step 2/4: Setting up dataset ===")
        try:
            import setup_dataset
            dataset_path = setup_dataset.setup_dataset()
            print(f"Dataset setup complete at: {dataset_path}")
            return True
        except Exception as e:
            print(f"Error setting up dataset: {str(e)}")
            return False

    def train_model(self):
        """Step 3: Train the ViT model"""
        print("\n=== Step 3/4: Training model ===")
        try:
            import train_vit
            train_vit.main()
            print("Model training complete!")
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False

    def launch_ui(self):
        """Step 4: Launch the Streamlit UI"""
        print("\n=== Step 4/4: Launching UI ===")
        try:
            # Check if model exists
            model_path = Path("./vit-xray-output/final_model")
            if not model_path.exists():
                print("Error: Model not found. Please train the model first.")
                return False

            # Launch Streamlit app
            print("Launching Streamlit UI...")
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", "predict.py"])
            print("UI launched successfully!")
            return True
        except Exception as e:
            print(f"Error launching UI: {str(e)}")
            return False

    def run(self):
        """Run the entire pipeline"""
        print("Starting X-Ray Analysis Pipeline...")
        
        for step in self.steps:
            self.current_step += 1
            print(f"\nProgress: {self.current_step}/{self.total_steps}")
            
            if not step():
                print(f"Pipeline failed at step {self.current_step}")
                return False
            
            time.sleep(1)  # Small delay between steps
        
        print("\nPipeline completed successfully!")
        return True

if __name__ == "__main__":
    pipeline = XRayPipeline()
    pipeline.run() 