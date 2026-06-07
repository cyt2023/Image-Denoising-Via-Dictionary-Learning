# Image Denoising Via Dictionary Learning

This project implements a classical sparse representation pipeline for image denoising:

- Gaussian baseline
- DCT dictionary + OMP baseline
- Learned dictionary using K-SVD + OMP

The code is designed for a signal processing course project and reproduces the figures and tables needed for a short report and presentation.

## Expected dataset layout

Place the Berkeley Segmentation Dataset images inside the project folder like this:

```text
BSDS300/
├── images/
│   ├── train/
│   └── test/
├── iids_train.txt
└── iids_test.txt
```

Only the images are used. Segmentation labels and benchmark code are not used.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the experiment

Quick validation run:

```bash
python main.py --fast_mode
```

Full experiment:

```bash
python main.py
```

You can also override parameters, for example:

```bash
python main.py --data_dir BSDS300/images/train --n_images 3 --image_size 256 --patch_size 8 --n_train_patches 6000 --n_atoms 256 --sparsity 6 --ksvd_iter 10 --seed 42 --output_dir results
```

## Generated outputs

The experiment automatically creates:

- `results/images/`
- `results/figures/`
- `results/tables/`
- `results/dictionaries/`

Main outputs:

- `results/tables/metrics.csv`
- `results/tables/metrics_summary.csv`
- comparison figures for each image and noise level
- average MSE vs noise variance plot
- average PSNR vs noise variance plot
- learned dictionary atom visualisations
- `results/selected_images.txt`

## Reproducing report figures

1. Run `python main.py` for the full experiment.
2. Use the comparison figures in `results/figures/` to show visual denoising quality.
3. Use `metrics_summary.csv` for average quantitative comparisons across methods and noise levels.
4. Use the metric-vs-noise plots to discuss robustness as noise variance increases.
5. Use the dictionary visualisations in `results/dictionaries/` to illustrate what K-SVD learns.
