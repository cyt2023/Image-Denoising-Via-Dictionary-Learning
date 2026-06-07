from __future__ import annotations

import numpy as np


def omp_single(D: np.ndarray, y: np.ndarray, sparsity: int) -> np.ndarray:
    _, n_atoms = D.shape
    x = np.zeros(n_atoms, dtype=np.float64)
    residual = y.astype(np.float64).copy()
    support: list[int] = []

    for _ in range(min(sparsity, n_atoms)):
        correlations = D.T @ residual
        if support:
            correlations[np.asarray(support)] = 0.0
        atom_idx = int(np.argmax(np.abs(correlations)))
        if atom_idx in support or abs(correlations[atom_idx]) < 1e-12:
            break
        support.append(atom_idx)

        D_support = D[:, support]
        coeffs, _, _, _ = np.linalg.lstsq(D_support, y, rcond=None)
        residual = y - D_support @ coeffs
        if np.linalg.norm(residual) < 1e-6:
            break

    if support:
        x[np.asarray(support)] = coeffs
    return x


def omp_batch(D: np.ndarray, Y: np.ndarray, sparsity: int) -> np.ndarray:
    n_atoms = D.shape[1]
    n_samples = Y.shape[1]
    X = np.zeros((n_atoms, n_samples), dtype=np.float64)
    for idx in range(n_samples):
        X[:, idx] = omp_single(D, Y[:, idx], sparsity)
    return X
