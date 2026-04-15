"""
app.py

Responsive UI version of SVD Image Compression Tool
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from svd_utils import (
    compress_image,
    calculate_compression_ratio,
    reconstruction_error
)


# ---------------- Page Config ----------------

st.set_page_config(
    page_title="SVD Image Compression Tool",
    page_icon="📊",
    layout="wide"
)


# ---------------- Title Section ----------------

st.title("SVD Image Compression Tool")

st.caption(
    "Interactive visualization of low-rank image approximation using Singular Value Decomposition (SVD)"
)


# ---------------- Upload Section ----------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Resize large images for performance
    MAX_SIZE = 512
    if max(image.size) > MAX_SIZE:
        image.thumbnail((MAX_SIZE, MAX_SIZE))

    image_array = np.array(image)


    # ---------------- Rank Selection ----------------

    st.markdown("### Compression Control")

    max_rank = min(image_array.shape[0], image_array.shape[1])
    slider_max = min(max_rank, 200)

    k = st.slider(
        "Select compression rank (k)",
        min_value=1,
        max_value=slider_max,
        value=min(50, slider_max)
    )


    # ---------------- Compression ----------------

    with st.spinner("Compressing image using SVD..."):
        compressed_image = compress_image(image_array, k)


    # ---------------- Image Comparison ----------------

    st.markdown("### Image Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image_array,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            compressed_image,
            caption=f"Compressed Image (k = {k})",
            use_container_width=True
        )


    # ---------------- Metrics ----------------

    st.markdown("### Compression Metrics")

    ratio = calculate_compression_ratio(image_array.shape, k)
    error = reconstruction_error(image_array, compressed_image)

    m1, m2 = st.columns(2)

    m1.metric(
        label="Compression Ratio",
        value=f"{ratio}% saved"
    )

    m2.metric(
        label="Reconstruction Error",
        value=f"{error}"
    )


    # ---------------- Graph Section ----------------

    st.markdown("### Rank vs Compression Analysis")

    ranks = list(range(5, slider_max + 1, 10))

    compression_ratios = []
    reconstruction_errors = []

    for r in ranks:
        temp = compress_image(image_array, r)

        compression_ratios.append(
            calculate_compression_ratio(image_array.shape, r)
        )

        reconstruction_errors.append(
            reconstruction_error(image_array, temp)
        )


    fig, ax = plt.subplots(1, 2)

    fig.set_size_inches(10, 4)


    ax[0].plot(ranks, compression_ratios)
    ax[0].set_title("Compression Ratio vs Rank")
    ax[0].set_xlabel("Rank (k)")
    ax[0].set_ylabel("Compression Ratio (%)")


    ax[1].plot(ranks, reconstruction_errors)
    ax[1].set_title("Reconstruction Error vs Rank")
    ax[1].set_xlabel("Rank (k)")
    ax[1].set_ylabel("Error")


    st.pyplot(fig, use_container_width=True)


    # ---------------- Download Section ----------------

    st.markdown("### Download Result")

    compressed_pil = Image.fromarray(compressed_image)

    buffer = BytesIO()
    compressed_pil.save(buffer, format="PNG")

    st.download_button(
        label="Download Compressed Image",
        data=buffer.getvalue(),
        file_name="compressed_image.png",
        mime="image/png"
    )