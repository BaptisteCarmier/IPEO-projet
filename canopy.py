from dataloader import *
from covnet import *

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import time

# Training setting

number_bands = 12
data_augmentation = True

# Hyperparameters
criterion = nn.SmoothL1Loss(reduction='none')
list_lr = [1e-3]
list_weight_decay = [1e-5]

# Results csv file

augmentation_status = "Aug" if data_augmentation else "NotAug"
csv_file = f"results_{number_bands}bands_{augmentation_status}.csv"

# Number of 255 filtering, comparison
number_filtered_outside = 0

# Function definition

def trainin_epochs(train_loader,optimizer,criterion,model,device="cuda"):
    losses =[]
    number_filtered_training = 0
    for batch_idx, (images, masks) in enumerate(train_loader):
        
        model.train()
        optimizer.zero_grad()

        # Move data to device (e.g., CUDA if available)
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)

        # Compute loss with filtering
        loss_map = criterion(outputs, masks)
        valid_mask = (masks != 255).float()
        valid_loss_map = loss_map*valid_mask
        num_valid_pixels = valid_mask.sum()        
        loss = valid_loss_map.sum() / num_valid_pixels

        # Validation: Check filtering stats
        total_pixels = masks.numel()
        valid_pixels = num_valid_pixels.item()
        non_valid_mask = (masks == 255).sum().item()
        print(f"Batch {batch_idx}: Total Pixels = {total_pixels}, Valid Pixels = {valid_pixels}, Filtered Pixels = {non_valid_mask}")
        number_filtered_training = number_filtered_training + non_valid_mask
        
        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Print loss for monitoring
        if batch_idx % 4 == 0:  # Print every 4 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        losses.append(loss.item())

    print("End of epoch : number of pixels filtered (with value = 255) inside the training loop = ", number_filtered_training)

    # Return mean loss
    mean_loss = np.mean(losses)

    # Sqrt(mean-loss)
    RootLoss_loss = np.sqrt(mean_loss)
    return mean_loss, RootLoss_loss

def validation_tot(validation_loader,model,criterion,device="cuda"):
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(validation_loader):

            # move to device
            images = images.to(device)
            masks = masks.to(device)

            # forward
            outputs = model(images)

            # filter out
            loss_map = criterion(outputs, masks)
            valid_mask = (masks != 255).float()
            valid_loss_map = loss_map*valid_mask
            num_valid_pixels = valid_mask.sum()        
            loss = valid_loss_map.sum() / num_valid_pixels
            val_loss+=loss.item()

            # Print loss for monitoring
            if batch_idx % 4 == 0:  # Print every 4 batches
                print(f"Validation step, Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    val_loss /= len(validation_loader)
    val_RootLoss = torch.sqrt(torch.tensor(val_loss)).item()
    return val_loss,val_RootLoss

# Data loading

RGB_bool = (number_bands == 3) # Bool = true if it's RGB

if data_augmentation == True:

    transforms_train = TransformCustom12(augmentations=[
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotation(angles=(90,180,270), p=0.5)
    ])

    train_dl = Sentinel2(csv_path=csv_path, split="train",transform=transforms_train, RGB = RGB_bool)
else:
    train_dl = Sentinel2(csv_path=csv_path, split="train", RGB = RGB_bool)

train_loader = DataLoader(train_dl, batch_size=256, shuffle=True)

validation_dl = Sentinel2(csv_path=csv_path, split="validation", RGB = RGB_bool)
validation_loader = DataLoader(validation_dl, batch_size=256, shuffle=False)

# Check cuda

assert torch.cuda.is_available()

print(f"--- Creating csv file -----------------------------")

# Check if the CSV file exists. If not, create a new one with headers
if not os.path.exists(csv_file):
    # Define the headers for the CSV file
    headers = ["Learning Rate", "Weight Decay", "Epoch", "Train Loss", "Train RootLoss", "Val Loss", "Val RootLoss", "Time Epoch"]
    # Create the CSV file and write the headers
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()  # Write the header row

def log_to_csv(lear, wd, epoch, trainloss, trainRootLoss, val_loss, val_RootLoss, time_epoch):
    # Log the values into the CSV
    data = {
        "Learning Rate": lear,
        "Weight Decay": wd,
        "Epoch": epoch,
        "Train Loss": trainloss,
        "Train RootLoss": trainRootLoss,
        "Val Loss": val_loss,
        "Val RootLoss": val_RootLoss,
        "Time Epoch": time_epoch
    }
    # Append the data to the CSV file
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writerow(data)

print("--- TRAINING ---------------------------------------")

# Training loop
num_epochs = 50 
for lear in list_lr:
    for wd in list_weight_decay:

        # Instantiate a new model to have new weights.
        model = UNetRegressor(in_channels=number_bands, out_channels=1)
        device = 'cuda'
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lear, weight_decay=wd)
        best_val_loss = float('inf')  # Initialize best validation loss


        for epoch in range(num_epochs):
            start_time = time.time()  # Start timing

            trainloss, trainRootLoss = trainin_epochs(train_loader,optimizer,criterion,model)
            val_loss,val_RootLoss = validation_tot(validation_loader,model,criterion)

            end_time = time.time()  # End timing
            epoch_time = end_time - start_time

            print("--- EPOCH n° ",epoch," ------------------------")
            print(f"Time taken: {epoch_time:.2f} seconds")
            print("trainloss,trainRootLoss:",trainloss,",",trainRootLoss)
            print("valloss,valRootLoss:",val_loss,",",val_RootLoss,"\n")
            log_to_csv(lear, wd, epoch, trainloss, trainRootLoss, val_loss, val_RootLoss, epoch_time)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_save_path = f"model_{number_bands}bands_{augmentation_status}_lr{lear}_wd{wd}.pth"
                print(f"New best model saved to {model_save_path} with val_loss: {val_loss:.4f}")
                torch.save(model.state_dict(), model_save_path)

print("--- END --------------------------------------")