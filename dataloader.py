## Function to unzip data file ##
import zipfile
with zipfile.ZipFile("canopy_height_dataset.zip", 'r') as zip_ref:
    zip_ref.extractall()

