from dataloader import *
from covnet import *
from metric import *

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Créer le DataLoader
train_dl = Sentinel2(csv_path=csv_path, split="train",RGB=False)
train_loader = DataLoader(train_dl, batch_size=256, shuffle=True)

# Test du DataLoader
for batch_idx, (images, masks) in enumerate(train_loader):
    print(f"Batch {batch_idx}")
    print(f"Images shape: {images.shape}")  # [batch_size, C, H, W]
    print(f"Masks shape: {masks.shape}")    # [batch_size, 1, H, W]
    break

assert torch.cuda.is_available()

# Instantiate the model
model = resnet101(pretrained = True)
device = 'cuda'
model = model.to(device)
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1))

# Test with dummy input :)
"""
dummy_input = torch.randn(2,12,32,32)
output = model(dummy_input)
print("Output shape:", output.shape)
"""

# Define loss and optimizer
criterion = nn.MSELoss(reduction = 'none')  # Or SmoothL1Loss

validation_dl = Sentinel2(csv_path=csv_path, split="validation")
validation_loader = DataLoader(validation_dl, batch_size=16, shuffle=False) #False pour permettre une comparaison entre les modèles

val_loss = 0.0

model.eval()
with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(validation_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)
        val_loss+=loss.item()

val_loss /= len(validation_loader)
print(f"Validation Loss: {val_loss:.4f}")

rmse = torch.sqrt(torch.tensor(val_loss))
print(f"Validation rmse: {rmse:.4f}")