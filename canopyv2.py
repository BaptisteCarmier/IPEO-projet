from dataloader import *
from covnet import *

import torch.nn as nn
import torch.optim as optim


def trainin_epochs(train_loader,optimizer,criterion,model,device="cuda"):
    losses =[]
    for batch_idx, (images, masks) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        # Move data to device (e.g., CUDA if available)
        images = images.to(device)
        masks = masks.to(device)
        # Forward pass
        outputs = model(images)
        # Compute loss
        loss = criterion(outputs, masks)
        # Backpropagation
        loss.backward()
        # Update weights
        optimizer.step()
        losses.append(loss.item())
    return np.stack(losses).mean(),torch.sqrt(torch.tensor(np.stack(losses).mean()))



def validation_tot(validation_loader,model,criterion,device="cuda"):
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
    #print(f"Validation Loss: {val_loss:.4f}")
    val_rmse = torch.sqrt(torch.tensor(val_loss))
    #print(f"Validation rmse: {rmse:.4f}")
    return val_loss,val_rmse


# Créer le DataLoader
train_dl = Sentinel2(csv_path=csv_path, split="train")
train_loader = DataLoader(train_dl, batch_size=16, shuffle=True)

############ validation dataset
validation_dl = Sentinel2(csv_path=csv_path, split="validation")
validation_loader = DataLoader(validation_dl, batch_size=16, shuffle=False) #False pour permettre une comparaison entre les modèles
'''
# Test du DataLoader
for batch_idx, (images, masks) in enumerate(train_loader):
    print(f"Batch {batch_idx}")
    print(f"Images shape: {images.shape}")  # [batch_size, C, H, W]
    print(f"Masks shape: {masks.shape}")    # [batch_size, 1, H, W]
    break
'''
assert torch.cuda.is_available()

# Instantiate the model
model = UNetRegressor(in_channels=12, out_channels=1)
device = 'cuda'
model = model.to(device)

# Test with dummy input :)
"""
dummy_input = torch.randn(2,12,32,32)
output = model(dummy_input)
print("Output shape:", output.shape)
"""

# Define loss and optimizer
criterion = nn.MSELoss()  # Or SmoothL1Loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training loop
num_epochs = 50  # Start with a small number of epochs to test
for epoch in range(num_epochs):
    trainloss, trainrmse = trainin_epochs(train_loader,optimizer,criterion,model)
    val_loss,val_rmse = validation_tot(validation_loader,model,criterion)
    print("For the epoch :",epoch)
    print("trainloss,trainrmse:",trainloss,",",trainrmse)
    print("valloss,valrmse:",val_loss,",",val_rmse,"\n")