import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from model import unet_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose

def load_images_and_masks(image_dir, mask_dir, target_size=(128, 128)):
    image_files = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])

    images = []
    masks = []

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Warning: Could not read image at {img_path}")
            continue
        if mask is None:
            print(f"Warning: Could not read mask at {mask_path}")
            continue

        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size)

        # Normalize image and mask
        image = image.astype('float32') / 255.0
        mask = (mask > 127).astype('float32')  # binary mask: 0 or 1
        mask = np.expand_dims(mask, axis=-1)

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load data
images, masks = load_images_and_masks(
    "/Users/vishaalg/Downloads/Brain Tumer Segmentation/images",
    "/Users/vishaalg/Downloads/Brain Tumer Segmentation/masks"
)

# Split data
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Build and compile model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=1, batch_size=8, validation_data=(X_val, y_val))

# Save model
model.save('/Users/vishaalg/Downloads/brain_tumor_segmentation_project/brain_tumor_unet.h5')

