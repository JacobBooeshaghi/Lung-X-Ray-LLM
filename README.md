# Lung-X-Ray-LLM
BME3503c Project

## Project Overview
This project implements a Vision Transformer (ViT) model for detecting pneumonia in chest X-ray images. The system includes a complete pipeline from dataset preparation to model training and a user-friendly interface for predictions.

## Project Structure

### Core Components

1. **setup_dataset.py**
   - Downloads and prepares the chest X-ray dataset from Kaggle
   - Handles API authentication and data organization
   - Returns the dataset path for use in other components

2. **dataloaders.py**
   - Implements data loading and preprocessing
   - Provides balanced dataset sampling
   - Handles data augmentation for training
   - Returns train, validation, and test dataloaders

3. **train_vit.py**
   - Implements the Vision Transformer model training
   - Uses Hugging Face's ViT implementation
   - Handles model training, validation, and evaluation
   - Generates attention maps and performance metrics

4. **predict.py**
   - Streamlit-based web interface for predictions
   - Handles image upload and preprocessing
   - Displays prediction results with confidence scores
   - Visualizes attention maps for model interpretability

5. **evaluation.py**
   - Contains evaluation metrics and visualization tools
   - Generates confusion matrices
   - Plots training history
   - Provides detailed classification reports

6. **pipeline.py**
   - Automates the entire workflow
   - Handles dataset setup, model training, and UI launch
   - Provides progress tracking and error handling
   - Simplifies the user experience

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Setup Kaggle API Key:
   - Go to your Kaggle account settings (https://www.kaggle.com/settings)
   - Scroll down to the "API" section
   - Click "Create New API Token" to download your `kaggle.json` file
   - Place the downloaded `kaggle.json` file in the project root directory

3. Run the complete pipeline:
```bash
python pipeline.py
```

This will:
- Setup your Kaggle API key
- Download and prepare the dataset
- Train the model
- Launch the web interface

## Manual Usage

If you prefer to run components separately:

1. Setup Kaggle API Key:
   - Follow the steps above to get your `kaggle.json`
   - Place it in the project root directory

2. Setup dataset:
```bash
python setup_dataset.py
```

3. Train model:
```bash
python train_vit.py
```

4. Launch UI:
```bash
streamlit run predict.py
```

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- Streamlit
- Kaggle API credentials (kaggle.json)

## Notes
- The trained model is too large to upload directly to GitHub
- You must have a Kaggle account and API key to download the dataset
- Training requires significant computational resources (GPU recommended)
- The model is for educational purposes only and should not be used for medical diagnosis

## Performance
The model achieves high accuracy in pneumonia detection, with detailed performance metrics available in the evaluation output. The system includes attention maps to help understand the model's decision-making process.

## Dataset
The project uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

