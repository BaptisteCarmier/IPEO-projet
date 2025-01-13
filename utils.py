import numpy as np
import matplotlib.pyplot as plt

def show_mask(pred1, pred2, pred3, pred4, pred5, mask):
    plt.figure(figsize=(12, 3))

    pred1 = pred1.squeeze(0)
    # Affichage de la pred1 en niveaux de gris
    plt.subplot(1,6,1)
    plt.imshow(pred1, cmap='gray')
    plt.colorbar(label="Pixel value", shrink=0.4)
    plt.title("Prediction 1")
    plt.axis("off")

    pred2 = pred2.squeeze(0)
    # Affichage de la pred2 en niveaux de gris
    plt.subplot(1,6,2)
    plt.imshow(pred2, cmap='gray')
    plt.colorbar(label="Pixel value", shrink=0.4)
    plt.title("Prediction 2")
    plt.axis("off")

    pred3 = pred3.squeeze(0)
    # Affichage de la pred3 en niveaux de gris
    plt.subplot(1,6,3)
    plt.imshow(pred3, cmap='gray')
    plt.colorbar(label="Pixel value", shrink=0.4)
    plt.title("Prediction 3")
    plt.axis("off")

    pred4 = pred4.squeeze(0)
    # Affichage de la pred4 en niveaux de gris
    plt.subplot(1,6,4)
    plt.imshow(pred4, cmap='gray')
    plt.colorbar(label="Pixel value", shrink=0.4)
    plt.title("Prediction 4")
    plt.axis("off")

    pred5 = pred5.squeeze(0)
    # Affichage de la pred5 en niveaux de gris
    plt.subplot(1,6,5)
    plt.imshow(pred5, cmap='gray')
    plt.colorbar(label="Pixel value", shrink=0.4)
    plt.title("Prediction 5")
    plt.axis("off")

    mask = mask.squeeze(0)
    # Affichage de la mask en niveaux de gris
    plt.subplot(1,6,6)
    plt.imshow(mask, cmap='gray')
    plt.colorbar(label="Pixel value", shrink=0.4)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def show_image_histogram_mask(image, mask):
    """
    Affiche l'image et son histogramme des valeurs de pixels pour les bandes RGB (1, 2, 3).
    """
    # Slicing pour ne garder que les bandes BGR (bandes 1, 2, 3)
    rgb_image = image[[3,2,1]]  # Réassigne pour avoir RGB et pas BGR

    # Normalisation des valeurs pour qu'elles soient dans [0, 1] pour imshow
    rgb_image = np.stack([(band - np.min(band)) / (np.max(band) - np.min(band)) for band in rgb_image])

    # Affichage de l'image RGB
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.imshow(rgb_image.transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)
    plt.title("RGB Image (Band 1, 2, 3)")
    plt.axis("off")

    # Affichage de l'histogramme des valeurs des pixels
    plt.subplot(1, 4, 2)
    plt.hist(rgb_image[0].reshape(-1), bins=50, alpha=0.7, label='B4-R', color='red')
    plt.hist(rgb_image[1].reshape(-1), bins=50, alpha=0.6, label='B3-G', color='green')
    plt.hist(rgb_image[2].reshape(-1), bins=50, alpha=0.5, label='B2-B', color='blue')
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of image pixel values (RGB)")
    plt.legend()

    mask = mask.squeeze(0)
    # Affichage du masque en niveaux de gris
    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap='gray')
    plt.colorbar(label="Pixel value")
    plt.title("Mask")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.hist(mask.reshape(-1), bins=50)
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.title("Histogram of mask pixel values")


    plt.tight_layout()
    plt.show()  

def show_before_after_augmentation(image_before, image_after, mask_before, mask_after):
    
    rgb_image_before = image_before[[3,2,1]]  
    rgb_image_before = np.stack([(band - np.min(band)) / (np.max(band) - np.min(band)) for band in rgb_image_before])

    rgb_image_after = image_after[[3,2,1]]  
    rgb_image_after = np.stack([(band - np.min(band)) / (np.max(band) - np.min(band)) for band in rgb_image_after])

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.imshow(rgb_image_before.transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)
    plt.title("RGB image before augmentation")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(rgb_image_after.transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)
    plt.title("RGB image after augmentation")
    plt.axis("off")

    mask_before = mask_before.squeeze(0)
    # Affichage du masque en niveaux de gris
    plt.subplot(1, 4, 3)
    plt.imshow(mask_before, cmap='gray')
    plt.colorbar(label="Pixel value")
    plt.title("Mask before augmentation")
    plt.axis("off")

    mask_after = mask_after.squeeze(0)
    # Affichage du masque en niveaux de gris
    plt.subplot(1, 4, 4)
    plt.imshow(mask_after, cmap='gray')
    plt.colorbar(label="Pixel value")
    plt.title("Mask after augmentation")
    plt.axis("off")

    
    plt.tight_layout()
    plt.show()  
    
