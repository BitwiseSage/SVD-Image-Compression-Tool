"""
svd_utils.py

Core mathematical functions for image compression using
Singular Value Decomposition (SVD).

This module handles:

1. Channel-wise compression using SVD
2. Full RGB image compression
3. Compression ratio estimation
4. Reconstruction error calculation (Frobenius norm)

Keeping math logic separate from UI makes the project clean,
modular, and easier to maintain.
"""

import numpy as np


def compress_channel(channel, k):
    """
    Compress a single image channel using rank-k approximation.

    SVD decomposes matrix A as:
        A = U * S * V^T

    Instead of storing full matrices, we keep only first k singular values:
        A_k = U_k * S_k * V_k^T

    This reduces storage while preserving important structure.

    Parameters
    ----------
    channel : numpy.ndarray
        2D array representing one color channel
    k : int
        Number of singular values to keep

    Returns
    -------
    numpy.ndarray
        Reconstructed compressed channel
    """

    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)

    # Keep only first k singular values/components
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    # Reconstruct compressed channel
    compressed_channel = U_k @ S_k @ Vt_k

    return compressed_channel


def compress_image(image, k):
    """
    Apply SVD compression separately on RGB channels.

    Since color images contain 3 channels (R, G, B),
    each channel is compressed independently and then recombined.

    Parameters
    ----------
    image : numpy.ndarray
        Input RGB image (height × width × 3)
    k : int
        Rank used for compression

    Returns
    -------
    numpy.ndarray
        Compressed RGB image
    """

    # Create empty array with same shape as original image
    compressed = np.zeros_like(image)

    # Compress each RGB channel separately
    for i in range(3):
        compressed[:, :, i] = compress_channel(image[:, :, i], k)

    # Clip values to valid pixel range
    compressed = np.clip(compressed, 0, 255)

    return compressed.astype(np.uint8)


def calculate_compression_ratio(image_shape, k):
    """
    Estimate storage saved after rank-k compression.

    Original storage:
        m × n × 3

    Compressed storage:
        k × (m + n + 1) × 3

    This function returns percentage storage reduction.

    Parameters
    ----------
    image_shape : tuple
        Shape of input image (height, width, channels)
    k : int
        Rank used for approximation

    Returns
    -------
    float
        Compression percentage
    """

    m, n, _ = image_shape

    original_storage = m * n * 3
    compressed_storage = k * (m + n + 1) * 3

    compression_percentage = (
        (1 - compressed_storage / original_storage) * 100
    )

    return round(compression_percentage, 2)


def reconstruction_error(original, compressed):
    """
    Compute reconstruction error using Frobenius norm.

    Formula:
        ||A - A_k||

    Lower error means better reconstruction quality.

    Parameters
    ----------
    original : numpy.ndarray
        Original image matrix
    compressed : numpy.ndarray
        Compressed image matrix

    Returns
    -------
    float
        Reconstruction error value
    """

    error = np.linalg.norm(original - compressed)

    return round(error, 2)