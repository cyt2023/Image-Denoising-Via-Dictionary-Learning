from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from src.ksvd import create_dct_dictionary, denoise_with_dictionary


def gaussian_filter_denoise(noisy_image: np.ndarray, sigma_noise: float) -> np.ndarray:
    filter_sigma = max(0.3, sigma_noise / 18.0)
    denoised = gaussian_filter(noisy_image, sigma=filter_sigma)
    return np.clip(denoised, 0.0, 255.0)


def dct_omp_denoise(
    noisy_image: np.ndarray,
    patch_size: int,
    sparsity: int,
    n_atoms: int,
    sigma_noise: float | None = None,
) -> np.ndarray:
    D = create_dct_dictionary(patch_size, n_atoms)
    return denoise_with_dictionary(
        noisy_image,
        D,
        patch_size,
        sparsity,
        sigma_noise=sigma_noise,
    )
