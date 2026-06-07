from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def extract_overlapping_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    windows = sliding_window_view(image, (patch_size, patch_size))
    return windows.reshape(-1, patch_size, patch_size).astype(np.float64)


def reconstruct_from_overlapping_patches(
    patches: np.ndarray,
    image_shape: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    height, width = image_shape
    out = np.zeros(image_shape, dtype=np.float64)
    weight = np.zeros(image_shape, dtype=np.float64)
    index = 0
    for row in range(height - patch_size + 1):
        for col in range(width - patch_size + 1):
            out[row : row + patch_size, col : col + patch_size] += patches[index]
            weight[row : row + patch_size, col : col + patch_size] += 1.0
            index += 1
    return out / np.maximum(weight, 1e-12)


def patches_to_matrix(patches: np.ndarray) -> np.ndarray:
    n_patches, patch_size, _ = patches.shape
    return patches.reshape(n_patches, patch_size * patch_size).T.astype(np.float64)


def matrix_to_patches(matrix: np.ndarray, patch_size: int) -> np.ndarray:
    return matrix.T.reshape(-1, patch_size, patch_size).astype(np.float64)


def sample_training_patches(
    patch_matrix: np.ndarray,
    n_train_patches: int,
    seed: int,
) -> np.ndarray:
    n_samples = patch_matrix.shape[1]
    if n_train_patches >= n_samples:
        return patch_matrix.copy()
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_samples, size=n_train_patches, replace=False)
    return patch_matrix[:, indices]
