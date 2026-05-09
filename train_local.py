import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import json
import os
import random

print("Loading insect pest dataset from HuggingFace (IP102 mirror) using streaming...")
dataset = load_dataset("EnmmmmOvO/insect-pest-dataset", split='train', streaming=True)

NUM_CLASSES = 30
SAMPLES_PER_CLASS = 50 # Keep it extremely small for fast local CPU training
VAL_SAMPLES_PER_CLASS = 10

print(f"Filtering dataset to {NUM_CLASSES} classes...")

# Manual collection of balanced subsets
train_items = []
val_items = []
class_counts = {i: 0 for i in range(NUM_CLASSES)}
val_counts = {i: 0 for i in range(NUM_CLASSES)}

# For IP102, we don't have the string class names natively in the dataset easily without metadata.
# We will just map them to "IP102_Pest_Class_0", etc. for this demo.
class_names = [f"IP102_Pest_Class_{i}" for i in range(NUM_CLASSES)]

for item in dataset:
    label = item['label']
    if label < NUM_CLASSES:
        if class_counts[label] < SAMPLES_PER_CLASS:
            train_items.append(item)
            class_counts[label] += 1
        elif val_counts[label] < VAL_SAMPLES_PER_CLASS:
            val_items.append(item)
            val_counts[label] += 1
            
    # Check if we have enough
    if all(c == SAMPLES_PER_CLASS for c in class_counts.values()) and all(v == VAL_SAMPLES_PER_CLASS for v in val_counts.values()):
        break

with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

print(f"Collected {len(train_items)} train images and {len(val_items)} validation images.")

class FastPestDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(FastPestDataset(train_items, transform_train), batch_size=32, shuffle=True)
val_loader = DataLoader(FastPestDataset(val_items, transform_val), batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading ResNet18...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 2 # Extremely short training for quick demo
best_acc = 0.0

print("Starting training (this should only take a few minutes)...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        print(f"  Batch {i+1}/{len(train_loader)}")
        
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")
    
    if val_acc >= best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_pest_model.pth')
        print("Saved best_pest_model.pth!")

print("Training complete! The FastAPI backend will automatically load the new model when restarted.")
