import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from dataloaders import get_dataloaders
from torchvision.transforms import Compose, Resize, ToTensor, Lambda, Normalize, transforms
from datasets import Dataset, load_dataset
import numpy as np
from PIL import Image

dataset = load_dataset("imagefolder", data_dir="/tmp/chest_xray/chest_xray/train")

# Show one sample to check the keys
print(dataset["train"][0])

dataset.set_format(type="python")  # Makes sure you get PIL images for the transform step
print(dataset["train"].features)

# transformers must be 4.38.0
# accelerate must be 0.27.2

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ViT
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2,
    id2label={0: "NORMAL", 1: "PNEUMONIA"},
    label2id={"NORMAL": 0, "PNEUMONIA": 1},
).to(device)

# Load the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Use your custom DataLoader
train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

# Helper to convert a PyTorch dataloader to a Hugging Face Dataset
def dataloader_to_hf_dataset(dataloader):
    images = []
    labels = []
    for x, y in dataloader:
        for i in range(len(x)):
            image = transforms.ToPILImage()(x[i])  # Convert tensor to PIL image
            images.append(image)
            labels.append(y[i].item())
    dataset = Dataset.from_dict({
        "image": images,  # Ensure key 'image' is present
        "label": labels
    })
    print(dataset[0])  # Debug: print first item to ensure it's correct
    return dataset

# Convert to Hugging Face Datasets
train_dataset = dataloader_to_hf_dataset(train_loader)
val_dataset = dataloader_to_hf_dataset(val_loader)

# Apply feature extractor
def transform(example_batch):
    images = [img.convert("RGB") for img in example_batch["image"]]
    return feature_extractor(images, return_tensors="pt")

dataset = dataset.map(transform, batched=True, batch_size=32, num_proc=1)
dataset["train"] = dataset["train"].map(transform, batched=True)
dataset["test"] = dataset["test"].map(transform, batched=True)

train_dataset.set_transform(transform)
val_dataset.set_transform(transform)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vit-xray-output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == labels)
    return {"accuracy": acc}

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

test_dataset = dataloader_to_hf_dataset(test_loader)
test_dataset.set_transform(transform)
metrics = trainer.evaluate(test_dataset)
print("Test accuracy:", metrics["eval_accuracy"])
