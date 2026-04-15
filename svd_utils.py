"""
svd_utils.py

Core SVD compression logic with proper float precision handling
to avoid reconstruction artifacts.
"""

import numpy as np


def compress_channel(channel, k):
    """
    Compress a single grayscale channel using rank-k SVD approximation.
    """

    # Convert to float for stable SVD computation
    channel = channel.astype(float)

    # Perform SVD decomposition
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)

    # Keep only first k singular values
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    # Reconstruct compressed channel
    compressed_channel = U_k @ S_k @ Vt_k

    return compressed_channel


def compress_image(image, k):
    """
    Apply SVD compression separately to RGB channels.
    """

    # Convert entire image to float BEFORE processing
    image = image.astype(float)

    compressed = np.zeros_like(image)

    for i in range(3):
        compressed[:, :, i] = compress_channel(image[:, :, i], k)

    # Clip safely AFTER reconstruction
    compressed = np.clip(compressed, 0, 255)

    # Convert back to uint8 only at the end
    return compressed.astype(np.uint8)


def calculate_compression_ratio(image_shape, k):
    """
    Estimate storage reduction percentage.
    """

    m, n, _ = image_shape

    original_storage = m * n * 3
    compressed_storage = k * (m + n + 1) * 3

    ratio = (1 - compressed_storage / original_storage) * 100

    return round(ratio, 2)


def reconstruction_error(original, compressed):
    """
    Compute Frobenius norm reconstruction error.
    """

    original = original.astype(float)
    compressed = compressed.astype(float)

    error = np.linalg.norm(original - compressed)

    return round(error, 2)