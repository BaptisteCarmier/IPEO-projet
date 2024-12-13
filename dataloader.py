import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio  # Remplacement de PIL par rasterio
import numpy as np
import matplotlib.pyplot

csv_path = "canopy_height_dataset/data_split.csv"

class Sentinel2(Dataset):
    def __init__(self, csv_path, split="train", transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file containing image/mask paths and splits.
            split (str): Dataset split to use ('train', 'val', 'test').
            transform (callable, optional): Transformations to apply to both images and masks.
        """
        # Load the CSV file
        self.data = pd.read_csv(csv_path, delimiter = ',')
        
        # Filter dataset based on the split
        self.data = self.data[self.data['split'] == split]
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image and label paths from the CSV
        row = self.data.iloc[idx]
        image_path = os.path.join("canopy_height_dataset/", row['image_path'])
        mask_path = os.path.join("canopy_height_dataset/", row['label_path'])
        
        # Load image with Rasterio
        with rasterio.open(image_path) as src:
            image = src.read()  # Lit toutes les bandes, shape: (C, H, W)
        
        # Charger le masque (supposant une seule bande)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Lit uniquement la première bande, shape: (H, W)
        
        # Convertir les types pour PyTorch
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
    
        
        # Gérer les valeurs nodata dans le masque
        mask[mask == 255] = -1  # Set nodata to -1 ou autre valeur si besoin
        print(mask)
        # Appliquer les transformations si elles existent
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # Convertir en tenseurs PyTorch
        image = torch.from_numpy(image)  # Pas besoin de permuter car Rasterio retourne (C, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # Ajouter la dimension du canal
        
        return image, mask


# Créer le DataLoader
train_dl = Sentinel2(csv_path=csv_path, split="train")
train_loader = DataLoader(train_dl, batch_size=16, shuffle=True)

# Test du DataLoader
for batch_idx, (images, masks) in enumerate(train_loader):
    print(f"Batch {batch_idx}")
    print(f"Images shape: {images.shape}")  # [batch_size, C, H, W]
    print(f"Masks shape: {masks.shape}")    # [batch_size, 1, H, W]
    break