import os
import numpy as np
import cv2

def load_images_and_masks(image_dir, mask_dir, target_size=(128, 128)):
    images = []
    masks = []
    
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir,"/Users/vishaalg/Downloads/Brain Tumer Segmentation/images")
        mask_path = os.path.join(mask_dir,"/Users/vishaalg/Downloads/Brain Tumer Segmentation/masks" )
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, target_size)
        image = image.astype('float32') / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_size)
        mask = np.expand_dims(mask, axis=-1)
        mask = (mask > 127).astype('float32')  # Binarize mask

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)
