# Canopy Height Regression

Project of Image Processign for Earth Observation - Ecole Polytechnique Fédérale de Lausanne (EPFL), Lausanne, Switzerland

## Team Members 
- Baptiste CARMIER
- Etienne DE LABARRIERE
- Maxime RISSE

## Project 

METRE L'ABSTRACT

# Requirement 
## Dataset 
A total of 9852 images (and the corresponding masks) has been used in this project. Each pair of image and mask is named with a unique id.
The dataset can be downloaded [here](https://enacshare.epfl.ch/bY2wS5TcA4CefGks7NtXg).


## Packages 
To run the code, a set of different package is necessary to use. Thus you should download it using 
```
pip install -r requirements.txt
```
Especially on anaconda it is recomended to create a new environnement with the depencedies listed in ```requirements.txt```. 

## GPU 
Please note that most of the code in this project are way more efficient on a GPU. It is also stated in some of them that they have to run on a GPU. Thus make sure that you have a GPU available on you computer. 


# Files in the folder

There are four files in the folder. 

- **README.md** :
The current file you're reading ;-).

- **requirements.txt**
Contains the diffrent packages to install.

- **report.pdf**
The scientific paper associated with this README.

- **Models**
A folder that contains 4 .pth files corresponding to the weight of 4 models we trained.

- **canopy.py** :
Contains the main code, and the training phase of this project.

- **dataloader.py**:
Contains the dataloader and is call by other scripts to load the data.

- **covnet.py** :
Contains the U-NET regressor architecture.

- **inference.ipynb**
Is a notbook showing the main results, such as a first import of the data, and prediction on a first batch of the test set.

- **plot___.py**
Are two script used to plot different graph during the training phase, the graph produced can be found in ```Results_1001_v3.pdf```.

- **utils.py**
Contains the different function used in ```ìnference.ipynb```.


