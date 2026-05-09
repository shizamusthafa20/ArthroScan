# Kaggle Full Training Script for IP102 (High Accuracy)
# INSTRUCTIONS:
# 1. Open https://www.kaggle.com/ and sign in.
# 2. Click "Create" on the left sidebar, then select "New Notebook".
# 3. On the right side panel under "Session options", change the "Accelerator" to "GPU T4x2" or "GPU P100".
# 4. Copy and paste this entire file into a code cell and click the Run button (Play icon).

import os
!pip install datasets huggingface_hub torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import json
import time

print("Loading full insect pest dataset from HuggingFace (IP102 mirror)...")
dataset = load_dataset("EnmmmmOvO/insect-pest-dataset")
train_dataset = dataset['train']
val_dataset = dataset['validation']

# We are training on 30 classes as requested
NUM_CLASSES = 30

def filter_data(example):
    return example['label'] < NUM_CLASSES

print(f"Filtering dataset to {NUM_CLASSES} classes...")
train_dataset = train_dataset.filter(filter_data)
val_dataset = val_dataset.filter(filter_data)

class_names = [
    "Rice Leaf Roller", "Rice Leaf Caterpillar", "Paddy Stem Maggot", "Asiatic Rice Borer", 
    "Yellow Rice Borer", "Rice Gall Midge", "Rice Stemfly", "Brown Plant Hopper", 
    "White Backed Plant Hopper", "Small Brown Plant Hopper", "Rice Water Weevil", 
    "Rice Leafhopper", "Grain Spreader Thrips", "Rice Shell Pest", "Grub", 
    "Mole Cricket", "Wireworm", "White Margined Moth", "Black Cutworm", 
    "Large Cutworm", "Yellow Cutworm", "Red Spider", "Corn Borer", 
    "Army Worm", "Aphids", "Potosiabre Vitarsis", "Peach Borer", 
    "English Grain Aphid", "Green Bug", "Bird Cherry-Oataphid"
]

with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

class FullPestDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Data Augmentation is crucial for >80% accuracy
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(FullPestDataset(train_dataset, transform_train), batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(FullPestDataset(val_dataset, transform_val), batch_size=64, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type != 'cuda':
    print("WARNING: You are not using a GPU. Training will take an extremely long time!")

print("Loading ResNet18...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

# Check for checkpoint in current folder OR in Kaggle input (if you uploaded it as a dataset)
checkpoint_path = 'best_pest_model.pth'
if not os.path.exists(checkpoint_path):
    # If you uploaded it as a Kaggle Dataset named 'pest-checkpoint', it will be here:
    input_path = '/kaggle/input/pest-checkpoint/best_pest_model.pth'
    if os.path.exists(input_path):
        checkpoint_path = input_path
    else:
        checkpoint_path = None

if checkpoint_path:
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Successfully loaded previous checkpoint from {checkpoint_path}! Resuming training.")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")

criterion = nn.CrossEntropyLoss()
# Using Cosine Annealing with Warm Restarts for better accuracy
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

num_epochs = 50
best_acc = 0.0

print(f"Starting training for {num_epochs} epochs on {device}...")
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    scheduler.step()
        
    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_acc = 100 * correct / total
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs} [{epoch_time:.0f}s] - Train Loss: {running_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f} - Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_pest_model.pth')
        print(f"  --> Saved new best model with {best_acc:.2f}% accuracy!")

print("Training complete! Download 'best_pest_model.pth' and 'class_names.json' from Colab.")
