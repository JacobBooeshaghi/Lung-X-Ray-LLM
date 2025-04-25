import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch.nn.functional as F
from transformers import ViTImageProcessor
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

def compute_metrics(predictions, labels):
    """
    Compute precision, recall, F1-score, and confusion matrix
    """
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    conf_matrix = confusion_matrix(labels, predictions)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

def plot_confusion_matrix(conf_matrix, save_path=None):
    """
    Plot and optionally save confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics
    """
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def generate_attention_map(model, image_path, save_path=None):
    """
    Generate and save attention visualization for a given image
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    inputs = processor(image, return_tensors="pt")
    
    # Get model attention weights
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attention = outputs.attentions[-1].mean(dim=1).mean(dim=1)  # Average over heads and batch
    
    # Create attention map
    attention_map = attention[0, 1:].reshape(14, 14).detach().numpy()  # Remove CLS token and reshape
    
    # Resize attention map to image size
    attention_map = cv2.resize(attention_map, (image.size[0], image.size[1]))
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    
    # Create heatmap overlay
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(attention_map, alpha=0.5, cmap='jet')
    plt.title('Attention Map')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def print_classification_report(y_true, y_pred):
    """
    Print detailed classification metrics
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA'])) 