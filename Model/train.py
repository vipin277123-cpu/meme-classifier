import sys
import os

# Fix import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Model.CNN import MemeCNN, MemeDataset, transform

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
data = pd.read_csv("Data/memes.csv")

# Split
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Dataset
train_dataset = MemeDataset(train_data, "Data/memes", transform)
test_dataset = MemeDataset(test_data, "Data/memes", transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model
model = MemeCNN().to(device)

# Loss (better for imbalance)
pos_weight = torch.tensor([2.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "meme_model.pt")
print("✅ Model Saved Successfully")