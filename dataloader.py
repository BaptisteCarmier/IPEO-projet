## Function to unzip data file ##
"""
import zipfile
with zipfile.ZipFile("canopy_height_dataset.zip", 'r') as zip_ref:
    zip_ref.extractall()
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import PIL
from PIL import Image
import numpy as np

from pathlib import Path

csv_path = "canopy_height_dataset\data_split.csv"

class Sentinel2(Dataset):
    def __init__(self, csv_path, split="train", transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file containing image/mask paths and splits.
            split (str): Dataset split to use ('train', 'val', 'test').
            transform (callable, optional): Transformations to apply to both images and masks.
        """
        # Load the CSV file
        self.data = pd.read_csv(csv_path)
        
        # Filter dataset based on the split
        self.data = self.data[self.data['split'] == split]
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image and label paths from the CSV
        row = self.data.iloc[idx]
        image_path = os.path.join("canopy_height_dataset/" + row['image_path'])
        mask_path = os.path.join("canopy_height_dataset/" + row['label_path'])
        
        # Load image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        
        # Convert to numpy arrays
        image = np.array(image, dtype=np.float32)  # Convert image to float32
        mask = np.array(mask, dtype=np.float32)    # Convert mask to float32
        
        # Handle nodata values in the mask
        mask[mask == 255] = -1  # Set nodata to -1 or any specific value as needed

        # Apply transforms if provided
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        mask = torch.from_numpy(mask).unsqueeze(0)        # Add channel dimension
        
        return image, mask

train_dl = Sentinel2(csv_path=csv_path, split="train")

train_loader = DataLoader(train_dl, batch_size=16, shuffle=True)

for batch_idx, (images, masks) in enumerate(train_loader):
    print(f"Batch {batch_idx}")
    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")
    break