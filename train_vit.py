import torch
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from dataloaders import get_dataloaders
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from evaluation import compute_metrics, plot_confusion_matrix, plot_training_history, generate_attention_map, print_classification_report
from torch.multiprocessing import freeze_support

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory
os.makedirs("output", exist_ok=True)

def main():
    # Load pretrained ViT model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,
        id2label={0: "NORMAL", 1: "PNEUMONIA"},
        label2id={"NORMAL": 0, "PNEUMONIA": 1},
    ).to(device)

    # Use ViTImageProcessor
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Get dataloaders with augmentation
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8,
        augment=True  # Enable data augmentation
    )

    class XRayDataset(torch.utils.data.Dataset):
        def __init__(self, dataloader, image_processor):
            self.images = []
            self.labels = []
            self.image_processor = image_processor
            
            # Extract all images and labels from dataloader
            for images, labels in dataloader:
                for i in range(len(images)):
                    self.images.append(transforms.ToPILImage()(images[i]))
                    self.labels.append(labels[i].item())
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            
            # Process image
            inputs = self.image_processor(image, return_tensors="pt")
            return {
                "pixel_values": inputs["pixel_values"].squeeze(),
                "labels": label
            }

    # Create datasets
    train_dataset = XRayDataset(train_loader, image_processor)
    val_dataset = XRayDataset(val_loader, image_processor)
    test_dataset = XRayDataset(test_loader, image_processor)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./vit-xray-output",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,  # Increased epochs for better convergence
        learning_rate=2e-5,   # Lower learning rate for fine-tuning
        warmup_ratio=0.1,     # Add warmup steps
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,   # Keep only the 2 best checkpoints
        weight_decay=0.01,    # Add weight decay for regularization
    )

    # Metrics computation function
    def compute_metrics_for_trainer(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        metrics = compute_metrics(predictions, labels)
        return {
            "accuracy": np.mean(predictions == labels),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_for_trainer,
    )

    # Train the model
    print("Starting training...")
    train_result = trainer.train()

    # Save the final model
    trainer.save_model("./vit-xray-output/final_model")
    print("Model saved to ./vit-xray-output/final_model")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    # Get predictions for confusion matrix
    test_predictions = trainer.predict(test_dataset)
    predictions = np.argmax(test_predictions.predictions, axis=1)
    true_labels = test_predictions.label_ids

    # Generate and save confusion matrix
    plot_confusion_matrix(
        compute_metrics(predictions, true_labels)["confusion_matrix"],
        save_path="./vit-xray-output/confusion_matrix.png"
    )

    # Print detailed classification report
    print_classification_report(true_labels, predictions)

    # Plot and save training history
    plot_training_history(history, save_path="./vit-xray-output/training_history.png")

    # Generate attention maps for a few test images
    print("\nGenerating attention maps for sample images...")
    test_image_dir = os.path.join(os.path.dirname(__file__), "data", "chest_xray", "test")
    for class_name in ["NORMAL", "PNEUMONIA"]:
        class_dir = os.path.join(test_image_dir, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir)[:3]:  # First 3 images of each class
                img_path = os.path.join(class_dir, img_name)
                save_path = f"./vit-xray-output/attention_map_{class_name}_{img_name}"
                generate_attention_map(model, img_path, save_path)

    print("\nTraining and evaluation completed!")
    print(f"Final test metrics: {test_results}")
    print("\nVisualization files saved in ./vit-xray-output directory")

if __name__ == '__main__':
    freeze_support()
    main()
