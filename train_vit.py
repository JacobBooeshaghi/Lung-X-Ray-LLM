import torch
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from dataloaders import get_dataloaders
from torchvision import transforms
from datasets import Dataset
import numpy as np
from PIL import Image

# Set device (CPU only)
device = torch.device("cpu")

# Load pretrained ViT model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2,
    id2label={0: "NORMAL", 1: "PNEUMONIA"},
    label2id={"NORMAL": 0, "PNEUMONIA": 1},
).to(device)

# Use ViTImageProcessor (newer than deprecated ViTFeatureExtractor)
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Get dataloaders
train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)

# Convert PyTorch DataLoader to Hugging Face Dataset
def dataloader_to_hf_dataset(dataloader):
    images = []
    labels = []
    for x, y in dataloader:
        for i in range(len(x)):
            image = transforms.ToPILImage()(x[i])
            images.append(image)
            labels.append(y[i].item())
    return Dataset.from_dict({"image": images, "label": labels})

train_dataset = dataloader_to_hf_dataset(train_loader)
val_dataset = dataloader_to_hf_dataset(val_loader)
test_dataset = dataloader_to_hf_dataset(test_loader)

# Define transform with image processor
def transform(example_batch):
    images = [img.convert("RGB") for img in example_batch["image"]]
    processed = image_processor(images, return_tensors="pt")
    return {
        "pixel_values": processed["pixel_values"],
        "label": example_batch["label"]
    }

# Set transform
train_dataset.set_transform(transform)
val_dataset.set_transform(transform)
test_dataset.set_transform(transform)

# TrainingArguments
training_args = TrainingArguments(
    output_dir="./vit-xray-output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Accuracy metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": np.mean(preds == labels)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on test set
metrics = trainer.evaluate(test_dataset)
print("Test accuracy:", metrics["eval_accuracy"])
