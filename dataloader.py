import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio  # Replacement of PIL by rasterio
import numpy as np


csv_path = "canopy_height_dataset/data_split.csv"

class Sentinel2(Dataset):
    def __init__(self, csv_path=csv_path, split="train", transform=None, RGB=False):
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
        self.RGB= RGB

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image and label paths from the CSV
        row = self.data.iloc[idx]
        image_path = os.path.join("canopy_height_dataset/", row['image_path'])
        mask_path = os.path.join("canopy_height_dataset/", row['label_path'])
        
        # Load image with Rasterio
        with rasterio.open(image_path) as src:
            image = src.read()  # Read all the bands, shape: (C, H, W)
        
        if self.RGB:
            image = image[[3,2,1]] #Select only BGR bands and put it in form RGB

        # Load the mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Read only the first band, shape: (H, W)
        
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        # Apply transforms if they exist
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Convert image and mask in PyTorch tensor
        image = torch.from_numpy(image)  
        mask = torch.from_numpy(mask).unsqueeze(0) 
        
        return image, mask


class TransformCustom12:
    def __init__(self, augmentations = None):
        self.augmentations = augmentations 

    def __call__(self, image, mask):
        for augment in self.augmentations :
            image, mask = augment(image, mask)
        
        return image, mask
    
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if np.random.rand()<self.p :
            image = np.flip(image, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()

        return image, mask
        
class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if np.random.rand()<self.p :
            image = np.flip(image, axis=-2).copy()
            mask = np.flip(mask, axis=-2).copy()

        return image, mask
    
class RandomRotation:
    def __init__(self, angles=(90,180,270), p=0.5):
        self.angles = angles
        self.p = p

    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            angle = np.random.choice(self.angles)
            if angle == 90:
                image = np.rot90(image, k=1, axes=(1, 2)).copy() 
                mask = np.rot90(mask, k=1, axes=(0, 1)).copy()
            elif angle == 180:
                image = np.rot90(image, k=2, axes=(1, 2)).copy()
                mask = np.rot90(mask, k=2, axes=(0, 1)).copy()
            elif angle == 270:
                image = np.rot90(image, k=3, axes=(1, 2)).copy()
                mask = np.rot90(mask, k=3, axes=(0, 1)).copy()
        return image, mask
