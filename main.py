from __future__ import annotations

import argparse

from src.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Image denoising via dictionary learning (K-SVD + OMP)."
    )
    parser.add_argument("--data_dir", default="BSDS300/images/train")
    parser.add_argument("--n_images", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--n_train_patches", type=int, default=6000)
    parser.add_argument("--n_atoms", type=int, default=256)
    parser.add_argument("--sparsity", type=int, default=6)
    parser.add_argument("--ksvd_iter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument(
        "--fast_mode",
        action="store_true",
        help="Use smaller settings for quick debugging runs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
