import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('/Users/vishaalg/Downloads/brain_tumor_segmentation_project/brain_tumor_unet.h5')
st.title("Brain Tumor Segmentation")

uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image_resized = image.resize((256, 256))
    img_array = np.expand_dims(np.expand_dims(np.array(image_resized) / 255.0, axis=0), axis=-1)
    
    prediction = model.predict(img_array)[0, :, :, 0]
    pred_mask = (prediction > 0.5).astype(np.uint8) * 255

    st.image([image, pred_mask], caption=["Original Image", "Predicted Tumor Mask"], width=256)
