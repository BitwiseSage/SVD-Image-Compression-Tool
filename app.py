"""
app.py

Streamlit interface for SVD Image Compression Tool

Features:

1. Upload image
2. Rank slider
3. Original vs compressed display
4. Compression ratio metric
5. Reconstruction error metric
6. Graphs:
   - Compression ratio vs rank
   - Reconstruction error vs rank
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from svd_utils import (
    compress_image,
    calculate_compression_ratio,
    reconstruction_error
)


# Title
st.title("SVD Image Compression Tool")


# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for performance
    MAX_SIZE = 512
    if max(image.size) > MAX_SIZE:
        image.thumbnail((MAX_SIZE, MAX_SIZE))

    image_array = np.array(image)


    # Show original
    st.subheader("Original Image")
    st.image(image_array, width=600)


    # Rank selection
    max_rank = min(image_array.shape[0], image_array.shape[1])
    slider_max = min(max_rank, 200)

    k = st.slider(
        "Select compression rank (k)",
        1,
        slider_max,
        min(50, slider_max)
    )


    # Compress image
    with st.spinner("Compressing image using SVD..."):
        compressed_image = compress_image(image_array, k)


    # Show compressed image
    st.subheader("Compressed Image")
    st.image(compressed_image, width=600)


    # Metrics
    ratio = calculate_compression_ratio(image_array.shape, k)
    error = reconstruction_error(image_array, compressed_image)

    st.subheader("Compression Analysis")

    st.write(f"Compression Ratio: **{ratio}% storage saved**")
    st.write(f"Reconstruction Error (Frobenius Norm): **{error}**")


    # -------- Graph Section --------

    st.subheader("Rank vs Compression Analysis Graphs")

    ranks = list(range(5, slider_max + 1, 10))

    compression_ratios = []
    reconstruction_errors = []

    for r in ranks:
        temp_compressed = compress_image(image_array, r)

        compression_ratios.append(
            calculate_compression_ratio(image_array.shape, r)
        )

        reconstruction_errors.append(
            reconstruction_error(image_array, temp_compressed)
        )


    # Plot graphs
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))


    # Compression ratio graph
    ax[0].plot(ranks, compression_ratios)
    ax[0].set_title("Compression Ratio vs Rank")
    ax[0].set_xlabel("Rank (k)")
    ax[0].set_ylabel("Compression Ratio (%)")


    # Reconstruction error graph
    ax[1].plot(ranks, reconstruction_errors)
    ax[1].set_title("Reconstruction Error vs Rank")
    ax[1].set_xlabel("Rank (k)")
    ax[1].set_ylabel("Error")


    st.pyplot(fig)