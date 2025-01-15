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

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 32 * 32),
    nn.Unflatten(1, (1, 32, 32))  # Reshape to correspond to (batch_size, 1, 32, 32)
)

model = model.to(device)

# Define loss and optimizer
criterion = nn.SmoothL1Loss(reduction = 'none')

test_dl = Sentinel2(csv_path=csv_path, split="test", RGB=True)
test_loader = DataLoader(test_dl, batch_size=32, shuffle=True) 

final_loss = 0.0

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

        final_loss += loss.item()
        
final_loss /= len(test_loader)
print(f"Test Loss: {final_loss:.4f}")

rmse = torch.sqrt(torch.tensor(final_loss))
print(f"Test rmse: {rmse:.4f}")