from dataloader import *
from covnet import *
from metric import *

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models

assert torch.cuda.is_available()

# Instantiate the model
model = models.resnet101(pretrained = True)
device = 'cuda'
# model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1))
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 32 * 32),
    nn.Unflatten(1, (1, 32, 32))  # Reshape pour correspondre à (batch_size, 1, 32, 32)
)

model = model.to(device)

# Test with dummy input :)
"""
dummy_input = torch.randn(2,12,32,32)
output = model(dummy_input)
print("Output shape:", output.shape)
"""

# Define loss and optimizer
criterion = nn.MSELoss(reduction = 'none')  # Or SmoothL1Loss zhre one we want

test_dl = Sentinel2(csv_path=csv_path, split="test", RGB=True)
test_loader = DataLoader(test_dl, batch_size=32, shuffle=True) #False pour permettre une comparaison entre les modèles

loss = 0.0

model.eval()
with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss_map = criterion(outputs, masks)
        valid_mask = (masks != 255).float()
        valid_loss_map = loss_map*valid_mask

        num_valid_pixels = valid_mask.sum()
        loss = valid_loss_map.sum() / num_valid_pixels

        #loss = criterion(outputs, masks)
        #val_loss+=loss.item() ## on a une erreur de taille ici RuntimeError: "a Tensor with 16384 elements cannot be converted to Scalar"

loss /= len(test_loader)
print(f"Test Loss: {loss:.4f}")

rmse = torch.sqrt(torch.tensor(loss))
print(f"Test rmse: {rmse:.4f}")