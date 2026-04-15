"""
app.py

Streamlit interface for SVD Image Compression Tool

New features added in this version:

1. Compression ratio display
2. Reconstruction error (Frobenius norm)
"""

import streamlit as st
import numpy as np
from PIL import Image

# Import functions from math module
from svd_utils import (
    compress_image,
    calculate_compression_ratio,
    reconstruction_error
)


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

    # Resize large images for faster SVD
    MAX_SIZE = 512
    if max(image.size) > MAX_SIZE:
        image.thumbnail((MAX_SIZE, MAX_SIZE))

    # Convert image to NumPy array
    image_array = np.array(image)

    # Display original image
    st.subheader("Original Image")
    st.image(image_array, width=600)


    # Determine max possible rank
    max_rank = min(image_array.shape[0], image_array.shape[1])

    # Limit slider range for performance
    slider_max = min(max_rank, 200)

    # Rank selection slider
    k = st.slider(
        "Select compression rank (k)",
        min_value=1,
        max_value=slider_max,
        value=min(50, slider_max)
    )


    # Compress image
    with st.spinner("Compressing image using SVD..."):
        compressed_image = compress_image(image_array, k)


    # Display compressed image
    st.subheader("Compressed Image")
    st.image(compressed_image, width=600)


    # Compute compression ratio
    ratio = calculate_compression_ratio(image_array.shape, k)

    # Compute reconstruction error
    error = reconstruction_error(image_array, compressed_image)


    # Display metrics
    st.subheader("Compression Analysis")

    st.write(f"Compression Ratio: **{ratio}% storage saved**")
    st.write(f"Reconstruction Error (Frobenius Norm): **{error}**")