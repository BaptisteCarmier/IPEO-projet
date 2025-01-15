import numpy as np
import matplotlib.pyplot as plt

def show_pred_mask(pred1, pred2, pred3, pred4, pred5, mask):

    images = [pred2, pred3, pred4, pred5, mask]
    vmin = None #min([img.min() for img in images]) #uncomment if you want the same color scale on each images
    vmax = None #max([img.max() for img in images]) #uncomment if you want the same color scale on each images

    valid_mask = (mask != 255)

    plt.figure(figsize=(12, 3))

    mae1 = np.sum(np.abs(pred1-mask)*valid_mask)/np.sum(valid_mask)
    mae2 = np.sum(np.abs(pred2-mask)*valid_mask)/np.sum(valid_mask)
    mae3 = np.sum(np.abs(pred3-mask)*valid_mask)/np.sum(valid_mask)
    mae4 = np.sum(np.abs(pred4-mask)*valid_mask)/np.sum(valid_mask)
    mae5 = np.sum(np.abs(pred5-mask)*valid_mask)/np.sum(valid_mask)


    pred1 = pred1.squeeze(0)
    plt.subplot(1,6,1)
    plt.imshow(pred1, cmap='Greens', vmin=vmin, vmax=vmax)
    plt.colorbar( shrink=0.4)
    plt.title("ResNet101")
    plt.text(0.5, -0.1, f"MAE : {mae1:.2f}", ha='center', va='center', fontsize=8,transform=plt.gca().transAxes, color='black')
    plt.axis("off")


    pred2 = pred2.squeeze(0)
    plt.subplot(1,6,2)
    plt.imshow(pred2, cmap='Greens', vmin=vmin, vmax=vmax)
    plt.colorbar( shrink=0.4)
    plt.title("12 bands DA")
    plt.text(0.5, -0.1, f"MAE : {mae2:.2f}", ha='center', va='center', fontsize=8,transform=plt.gca().transAxes, color='black')
    plt.axis("off")


    pred3 = pred3.squeeze(0)
    plt.subplot(1,6,3)
    plt.imshow(pred3, cmap='Greens', vmin=vmin, vmax=vmax)
    plt.colorbar( shrink=0.4)
    plt.title("12 bands no DA")
    plt.text(0.5, -0.1, f"MAE : {mae3:.2f}", ha='center', va='center', fontsize=8,transform=plt.gca().transAxes, color='black')
    plt.axis("off")


    pred4 = pred4.squeeze(0)
    plt.subplot(1,6,4)
    plt.imshow(pred4, cmap='Greens', vmin=vmin, vmax=vmax)
    plt.colorbar( shrink=0.4)
    plt.title("RGB DA")
    plt.text(0.5, -0.1, f"MAE : {mae4:.2f}", ha='center', va='center', fontsize=8,transform=plt.gca().transAxes, color='black')
    plt.axis("off")


    pred5 = pred5.squeeze(0)
    plt.subplot(1,6,5)
    plt.imshow(pred5, cmap='Greens', vmin=vmin, vmax=vmax)
    plt.colorbar( shrink=0.4)
    plt.title("RGB no DA")
    plt.text(0.5, -0.1, f"MAE : {mae5:.2f}", ha='center', va='center', fontsize=8,transform=plt.gca().transAxes, color='black')
    plt.axis("off")

    mask = mask.squeeze(0)
    plt.subplot(1,6,6)
    plt.imshow(mask, cmap='Greens', vmin=vmin, vmax=vmax)
    plt.colorbar( shrink=0.4)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def show_image_histogram_mask(image, mask):

    rgb_image = image[[3,2,1]]

    rgb_image = np.stack([(band - np.min(band)) / (np.max(band) - np.min(band)) for band in rgb_image])

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.imshow(rgb_image.transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)
    plt.title("RGB Image (Band 1, 2, 3)")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.hist(rgb_image[0].reshape(-1), bins=50, alpha=0.7, label='B4-R', color='red')
    plt.hist(rgb_image[1].reshape(-1), bins=50, alpha=0.6, label='B3-G', color='green')
    plt.hist(rgb_image[2].reshape(-1), bins=50, alpha=0.5, label='B2-B', color='blue')
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of image pixel values (RGB)")
    plt.legend()

    mask = mask.squeeze(0)
    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap='Greens')
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
    plt.subplot(1, 4, 3)
    plt.imshow(mask_before, cmap='Greens')
    plt.colorbar(label="Pixel value")
    plt.title("Mask before augmentation")
    plt.axis("off")

    mask_after = mask_after.squeeze(0)
    plt.subplot(1, 4, 4)
    plt.imshow(mask_after, cmap='Greens')
    plt.colorbar(label="Pixel value")
    plt.title("Mask after augmentation")
    plt.axis("off")

    
    plt.tight_layout()
    plt.show()  
    
