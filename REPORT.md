# Report Notes: Image Denoising via Dictionary Learning

## 1. Problem Description

This project studies image denoising using sparse dictionary learning. The task is to recover a clean grayscale image from an observation corrupted by zero-mean white Gaussian noise. The image is divided into small `8 x 8` patches, each of which is represented as a vector `y_k`. For each patch, the goal is to find a sparse coefficient vector `x_k` and a dictionary `A` such that:

```text
y_k = A x_k + n_k
```

where `n_k` is additive Gaussian noise. Once the dictionary and sparse coefficients are estimated, the denoised patch is reconstructed as `A x_k`, and the full image is obtained by averaging overlapping reconstructed patches.

## 2. Brief Literature Survey

Sparse coding assumes that natural image patches can be represented accurately using only a small number of atoms from an overcomplete dictionary. This makes it well suited for denoising because signal structure tends to be compressible, while additive Gaussian noise is not. As a result, sparse representations can preserve edges and textures while suppressing noise.

The most important reference for this project is the K-SVD paper by Aharon, Elad, and Bruckstein. K-SVD alternates between sparse coding and dictionary updates. In the sparse coding step, each patch is approximated using only a few atoms, commonly with Orthogonal Matching Pursuit. In the dictionary update step, each atom and its associated coefficients are refined using a rank-1 singular value decomposition of the current residual. This makes K-SVD a practical and widely used algorithm for learning overcomplete dictionaries from image patches.

Compared with a fixed transform such as DCT, a learned dictionary adapts to the structures present in the data. This is especially useful for natural images, where local patterns such as edges, corners, and textures vary from image to image. A learned dictionary can therefore provide sparser and more faithful representations than a universal handcrafted basis.

The later Bayesian dictionary learning papers by Fedorov and Rao and by Joseph and Murthy show that dictionary learning can also be cast in a probabilistic framework, where uncertainty in both coefficients and dictionary atoms is modeled explicitly. These approaches are theoretically appealing and can improve robustness, but they are also more complex to implement and analyze in a compact course project. For that reason, this project uses K-SVD as a strong classical baseline that remains closely aligned with the sparse representation model in the assignment.

## 3. Dataset and Experimental Setup

The Berkeley Segmentation Dataset is used as the image source. Three images are selected uniformly at random from the dataset and converted to grayscale if needed. To keep experiments comparable, each image is normalized to a fixed `256 x 256` working size by center cropping or resizing when necessary.

Independent Gaussian noise is added to each image with standard deviations:

- `sigma = 5`
- `sigma = 10`
- `sigma = 15`
- `sigma = 25`

Each noisy image is divided into overlapping `8 x 8` patches. This gives patch vectors of dimension `64`. For dictionary learning, `6000` patches are selected uniformly at random from the noisy image. These settings follow the assignment specification directly.

## 4. Model Design

Three denoising methods are compared.

### 4.1 Gaussian Filtering Baseline

Gaussian smoothing is included as a simple non-sparse baseline. It provides a reference point for how much denoising can be achieved with a standard low-pass filter. Its weakness is that it tends to blur edges and fine textures together with the noise.

### 4.2 DCT Dictionary + OMP

The second method uses a fixed overcomplete DCT dictionary and Orthogonal Matching Pursuit. This method preserves the sparse-coding framework while avoiding the complexity of dictionary learning. It therefore serves as a useful baseline for testing whether learning the dictionary actually improves denoising quality.

### 4.3 K-SVD + OMP

The final model uses K-SVD to learn a dictionary directly from randomly sampled noisy patches and then reconstructs all image patches using OMP. This design was chosen because:

- sparse patch models are well matched to natural-image structure
- K-SVD learns atoms adapted to the current image instead of relying on a fixed transform
- OMP is a standard and interpretable sparse coding algorithm for the K-SVD pipeline
- the method is directly supported by the main reference in the assignment

The dictionary is overcomplete, meaning the number of atoms is larger than the patch dimension. This gives the model enough flexibility to represent diverse local structures while still encouraging sparse coefficients.

## 5. Reconstruction Procedure

The denoising pipeline is:

1. Extract all overlapping `8 x 8` patches from the noisy image.
2. Center the patch vectors by subtracting their mean intensity.
3. Learn a dictionary from `6000` randomly sampled training patches.
4. Compute sparse coefficients for all patches using OMP.
5. Reconstruct each patch from the dictionary and sparse coefficients.
6. Add back the patch mean.
7. Average overlapping reconstructed patches to form the final denoised image.

The overlap averaging step is important because it reduces block artifacts and stabilizes reconstruction.

## 6. Why These Parameters Were Chosen

The final parameter choices are motivated by a balance between assignment compliance, interpretability, and runtime:

- `patch size = 8 x 8`: standard in sparse image denoising and explicitly required
- `6000` training patches: matches the assignment and provides enough variability for dictionary learning
- `256` atoms: gives an overcomplete dictionary relative to the `64`-dimensional patch space
- `sparsity = 6`: allows compact patch descriptions while keeping reconstruction flexible
- `10` K-SVD iterations: enough to update the dictionary meaningfully without excessive runtime

These values can be tuned further, but they are a reasonable operating point for a compact course project.

## 7. Assessment and Validation Procedure

The model is assessed both qualitatively and quantitatively.

### 7.1 Quantitative Metrics

Two metrics are used:

- `MSE`: measures average squared reconstruction error
- `PSNR`: measures reconstruction quality on a logarithmic scale that is standard in image denoising

MSE is useful because it gives a direct measure of estimation error. PSNR is included because it is widely used in image restoration and is easier to compare visually across methods and noise levels.

### 7.2 Validation Strategy

The validation procedure compares all methods under the same conditions:

- the same three randomly chosen images
- the same four noise levels
- the same patch size
- the same reconstruction image size

This controlled setup makes the comparison fair. The main validation question is whether the learned dictionary consistently improves the denoised output over both the noisy input and the simpler baselines.

### 7.3 Error as a Function of Noise Variance

The assignment asks to study the estimation error as a function of noise variance. This is done by evaluating each method at `sigma^2 = 25, 100, 225, 625` and plotting average `MSE` and `PSNR` over the selected images. These curves show not only which method performs best, but also how robust each method remains as the corruption becomes stronger.

## 8. Expected Discussion of Results

The expected trend is:

- noisy images deteriorate as noise variance increases
- Gaussian filtering reduces noise but also blurs detail
- DCT + OMP typically improves over Gaussian filtering by exploiting patch sparsity
- K-SVD + OMP should perform best in most cases because the learned dictionary adapts to the image content

If K-SVD does not outperform the fixed DCT baseline at every noise level, possible reasons include insufficient iterations, limited training patches, dictionary initialization effects, or the fact that the dictionary is learned from noisy rather than clean patches.

## 9. Limitations

This project is intentionally compact and has several limitations:

- only three images are used, so conclusions should be treated as illustrative rather than definitive
- the dictionary is learned independently for each image and noise level, which increases runtime
- the implementation uses a classical K-SVD pipeline rather than a Bayesian dictionary learning method
- results may vary slightly with the random seed and the sampled patches

These limitations are acceptable for a course assignment, but they should be acknowledged in the report.

## 10. Conclusion

This project provides a classical sparse representation approach to image denoising using K-SVD and OMP. It satisfies the main technical requirements of the assignment, offers clear baseline comparisons, and produces both quantitative and qualitative outputs suitable for analysis. The main remaining work for submission is to present the results clearly and relate them back to the sparse coding literature and model-design choices.

## References

1. M. Aharon, M. Elad, and A. Bruckstein, "K-SVD: An algorithm for designing overcomplete dictionaries for sparse representation," IEEE Transactions on Signal Processing, vol. 54, no. 11, pp. 4311-4322, Nov. 2006.
2. I. Fedorov and B. D. Rao, "Multimodal sparse Bayesian dictionary learning," arXiv:1804.03740, 2018.
3. G. Joseph and C. R. Murthy, "On the Convergence of a Bayesian Algorithm for Joint Dictionary Learning and Sparse Recovery," IEEE Transactions on Signal Processing, vol. 68, pp. 343-358, 2020.
