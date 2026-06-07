# Image Denoising Via Dictionary Learning

This project implements a classical sparse representation pipeline for grayscale image denoising and is structured to match the main experimental requirements of the course assignment:

- Gaussian filtering baseline
- Fixed DCT dictionary + OMP sparse coding baseline
- Learned dictionary using K-SVD + OMP

The repository contains the code needed to generate denoised images, quantitative error tables, and visualisations for a short report or presentation. A report-ready discussion of the literature, model design, and validation strategy is included in [REPORT.md](REPORT.md).

## Assignment Mapping

The current implementation covers the main technical requirements from the assignment brief:

- Randomly selects `3` grayscale images from the Berkeley Segmentation Dataset
- Adds zero-mean Gaussian noise with standard deviations `5`, `10`, `15`, and `25`
- Splits each image into overlapping `8 x 8` patches
- Samples `6000` patches to train the dictionary
- Learns a sparse representation model using `K-SVD + OMP`
- Evaluates denoising quality with `MSE` and `PSNR`
- Plots estimation error as a function of noise variance

Default experiment settings:

- `n_images = 3`
- `patch_size = 8`
- `n_train_patches = 6000`
- `noise levels = [5, 10, 15, 25]`

One implementation detail not explicitly stated in the assignment is image size normalisation. Images are center-cropped to `256 x 256` when possible, or resized to `256 x 256` for consistency across experiments.

## Expected Dataset Layout

Place the Berkeley Segmentation Dataset images inside the project folder like this:

```text
BSDS300/
├── images/
│   ├── train/
│   └── test/
├── iids_train.txt
└── iids_test.txt
```

Only the images are used. Segmentation labels and benchmark code are not required.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Experiment

Quick validation run:

```bash
python main.py --fast_mode
```

Full experiment with assignment-style defaults:

```bash
python main.py
```

Example with explicit parameters:

```bash
python main.py --data_dir BSDS300/images/train --n_images 3 --image_size 256 --patch_size 8 --n_train_patches 6000 --n_atoms 256 --sparsity 6 --ksvd_iter 10 --seed 42 --output_dir results
```

## Generated Outputs

The experiment automatically creates:

- `results/images/`
- `results/figures/`
- `results/tables/`
- `results/dictionaries/`

Main outputs:

- `results/tables/metrics.csv`
- `results/tables/metrics_wide.csv`
- `results/tables/metrics_summary.csv`
- comparison figures for each image and noise level
- average `MSE` vs noise variance plot
- average `PSNR` vs noise variance plot
- learned dictionary atom visualisations
- `results/selected_images.txt`

## Suggested Report Workflow

1. Run `python main.py` using the default settings.
2. Use the comparison figures in `results/figures/` for qualitative visual assessment.
3. Use `metrics_summary.csv` to compare methods quantitatively across noise levels.
4. Use the `MSE` and `PSNR` plots to discuss performance as noise variance increases.
5. Use dictionary atom visualisations to comment on what K-SVD learns beyond a fixed DCT basis.
6. Reuse the structure in [REPORT.md](REPORT.md) for the literature survey, method justification, and evaluation discussion.
