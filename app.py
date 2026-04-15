"""
app.py

Streamlit interface for SVD Image Compression Tool
"""

import streamlit as st
import numpy as np
from PIL import Image
from svd_utils import compress_image


# Page title
st.title("SVD Image Compression Tool")


# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Resize large images for faster SVD processing
    MAX_SIZE = 512
    if max(image.size) > MAX_SIZE:
        image.thumbnail((MAX_SIZE, MAX_SIZE))

    # Convert image to numpy array
    image_array = np.array(image)

    # Show original image
    st.subheader("Original Image")
    st.image(image_array, width=600)


    # Determine safe rank range
    max_rank = min(image_array.shape[0], image_array.shape[1])

    # Limit slider max for performance
    slider_max = min(max_rank, 200)

    # Compression slider
    k = st.slider(
        "Select compression rank (k)",
        min_value=1,
        max_value=slider_max,
        value=min(50, slider_max)
    )


    # Compress image
    with st.spinner("Compressing image using SVD..."):
        compressed_image = compress_image(image_array, k)


    # Show compressed image
    st.subheader("Compressed Image")
    st.image(compressed_image, width=600)