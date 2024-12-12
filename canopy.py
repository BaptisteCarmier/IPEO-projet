from dataloader import *
from module import *

import torch.nn as nn
import torch.optim as optim

# Créer le DataLoader
train_dl = Sentinel2(csv_path=csv_path, split="train")
train_loader = DataLoader(train_dl, batch_size=16, shuffle=True)

# Test du DataLoader
for batch_idx, (images, masks) in enumerate(train_loader):
    print(f"Batch {batch_idx}")
    print(f"Images shape: {images.shape}")  # [batch_size, C, H, W]
    print(f"Masks shape: {masks.shape}")    # [batch_size, 1, H, W]
    break


# Instantiate the model
model = UNetRegressor(in_channels=12, out_channels=1)

# Test with dummy input :)
dummy_input = torch.randn(2,12,32,32)
output = model(dummy_input)
print("Output shape:", output.shape)

# Define loss and optimizer
criterion = nn.MSELoss()  # Or SmoothL1Loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 1  # Start with a small number of epochs to test
for epoch in range(num_epochs):
    for batch_idx, (images, masks) in enumerate(train_loader):
        # Move data to device (e.g., CUDA if available)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, masks)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Print loss for monitoring
        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

print("Training is done ;)")