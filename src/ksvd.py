from __future__ import annotations

import math

import numpy as np
from tqdm import tqdm

from src.omp import omp_batch
from src.patch_utils import (
    extract_overlapping_patches,
    matrix_to_patches,
    patches_to_matrix,
    reconstruct_from_overlapping_patches,
)


def normalise_columns(D: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return D / norms


def initialise_dictionary(Y_train: np.ndarray, n_atoms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_samples = Y_train.shape[1]
    replace = n_samples < n_atoms
    indices = rng.choice(n_samples, size=n_atoms, replace=replace)
    D = Y_train[:, indices].copy()
    return normalise_columns(D)


def create_dct_dictionary(patch_size: int, n_atoms: int) -> np.ndarray:
    atoms_per_dim = math.ceil(math.sqrt(n_atoms))
    dct_1d = np.zeros((patch_size, atoms_per_dim), dtype=np.float64)
    grid = np.arange(patch_size, dtype=np.float64)

    for k in range(atoms_per_dim):
        basis = np.cos(grid * k * np.pi / atoms_per_dim)
        if k > 0:
            basis = basis - np.mean(basis)
        dct_1d[:, k] = basis / np.linalg.norm(basis)

    D = np.kron(dct_1d, dct_1d)
    D = D[:, :n_atoms]
    return normalise_columns(D)


def ksvd(
    Y_train: np.ndarray,
    n_atoms: int,
    sparsity: int,
    n_iter: int,
    seed: int,
    init_method: str = "random",
) -> tuple[np.ndarray, np.ndarray]:
    if init_method == "dct":
        patch_size = int(round(np.sqrt(Y_train.shape[0])))
        D = create_dct_dictionary(patch_size, n_atoms)
    else:
        D = initialise_dictionary(Y_train, n_atoms, seed)

    rng = np.random.default_rng(seed)
    X = np.zeros((n_atoms, Y_train.shape[1]), dtype=np.float64)

    for _ in tqdm(range(n_iter), desc="K-SVD", leave=False):
        X = omp_batch(D, Y_train, sparsity)

        for atom_idx in range(n_atoms):
            usage = np.flatnonzero(np.abs(X[atom_idx]) > 1e-12)
            if usage.size == 0:
                replacement_idx = int(rng.integers(0, Y_train.shape[1]))
                D[:, atom_idx] = Y_train[:, replacement_idx]
                D[:, atom_idx] /= max(np.linalg.norm(D[:, atom_idx]), 1e-12)
                continue

            coeffs = X[atom_idx, usage].copy()
            X[atom_idx, usage] = 0.0
            residual = Y_train[:, usage] - D @ X[:, usage]

            U, S, Vt = np.linalg.svd(residual, full_matrices=False)
            D[:, atom_idx] = U[:, 0]
            X[atom_idx, usage] = S[0] * Vt[0, :]

        D = normalise_columns(D)

    X = omp_batch(D, Y_train, sparsity)
    return D, X


def denoise_with_dictionary(
    noisy_image: np.ndarray,
    D: np.ndarray,
    patch_size: int,
    sparsity: int,
) -> np.ndarray:
    patches = extract_overlapping_patches(noisy_image, patch_size)
    patch_matrix = patches_to_matrix(patches)
    patch_means = np.mean(patch_matrix, axis=0, keepdims=True)
    centered = patch_matrix - patch_means
    codes = omp_batch(D, centered, sparsity)
    reconstructed = D @ codes + patch_means
    denoised_patches = matrix_to_patches(reconstructed, patch_size)
    denoised = reconstruct_from_overlapping_patches(
        denoised_patches, noisy_image.shape, patch_size
    )
    return np.clip(denoised, 0.0, 255.0)
