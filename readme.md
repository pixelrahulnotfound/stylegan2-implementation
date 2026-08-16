# StyleGAN2 in PyTorch

A clean, modular, and optimized PyTorch implementation of the **StyleGAN2** architecture for high-quality image generation. This repository contains the complete codebase for training the model from scratch, as well as running fast inference using the best pre-trained checkpoint weights.

This implementation successfully addresses key limitations of the original StyleGAN:
1. **No AdaIN Artifacts:** Replaces Adaptive Instance Normalization (AdaIN) with runtime weight modulation and demodulation, completely eliminating "water-droplet" artifacts.
2. **Smooth Training Without Progressive Growing:** Replaces progressive structure growing with skip connections in the Generator and residual connections in the Discriminator, ensuring stable training directly at the final resolution while resolving phase alignment artifacts (e.g., rigid teeth/eyes).
3. **Lazy Regularization:** Incorporates computationally efficient R1 regularization for the Discriminator and Perceptual Path Length (PPL) regularization for the Generator.

---

## Project Structure

```bash
├── models.py      # Core network components (Modulated Conv, Generator, Discriminator)
├── losses.py      # GAN loss functions, R1 regularization, and PPL regularization
├── train.py       # Full training pipeline with CLI options and FID logging
├── generate.py    # Image generation inference script with the truncation trick
├── invert.py      # GAN Inversion tool (project custom images into latent space)
├── weights/       # Directory containing pre-trained weights
│   └── best_checkpoint_1010k.pth
└── output/        # Default directory for generated images
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd gans
   ```

2. **Install dependencies:**
   Make sure you have PyTorch and Torchvision installed. You can install the remaining requirements via:
   ```bash
   pip install tqdm pillow numpy matplotlib torchmetrics
   ```

---

## Quick Start: Generating Images

You can generate images instantly using the pre-trained checkpoint weights included in the `weights/` directory.

### Basic Generation
Generate a grid of 16 images using default settings (recommends CUDA if available):
```bash
python3 generate.py
```

### Advanced Inference Options
Customize the generation using command line arguments:
```bash
python3 generate.py \
    --weights weights/best_checkpoint_1010k.pth \
    --output_dir output \
    --num_images 16 \
    --truncation 0.7 \
    --seed 42 \
    --device cuda
```

### Argument Reference:
* `--weights`: Path to the generator checkpoint.
* `--output_dir`: Directory to save generated images.
* `--num_images`: Number of images to generate (default: 16).
* `--truncation`: Truncation factor for the truncation trick (values between 0.0 and 1.0; default: 0.7. Lower values increase image quality and fidelity at the expense of variety).
* `--seed`: Set a random seed for reproducible image generation.
* `--grid`: Boolean flag. If `True` (default), saves all generated images as a single combined grid. If `False`, saves each image individually.
* `--device`: Force a specific device (e.g., `cuda`, `mps`, `cpu`).

---

## Training on a Custom Dataset

To train the model on your own dataset (e.g., FFHQ or custom faces):

1. **Prepare your dataset:**
   Organize your dataset folder in the PyTorch `ImageFolder` structure:
   ```bash
   dataset/
   └── class_name/
       ├── img1.png
       ├── img2.png
       └── ...
   ```

2. **Run the training script:**
   ```bash
   python3 train.py --data_path /path/to/dataset --batch_size 16 --lr 0.0025 --total_iters 1000000
   ```

### Training Argument Reference:
* `--data_path`: Path to the dataset root folder.
* `--checkpoints_dir`: Folder where checkpoints and image samples will be saved.
* `--batch_size`: Batch size (default: 16).
* `--lr`: Learning rate (default: 0.0025).
* `--latent_dim`: Latent space dimensionality (default: 512).
* `--total_iters`: Total training iterations.
* `--eval_freq`: Iteration interval for calculating FID and saving checkpoints.
* `--resume`: Path to a checkpoint `.pth` file to resume training.

---

## GAN Inversion (Image Projection)

To find the latent code $w$ that reconstructs a specific target face photo, you can run the GAN Inversion script. This optimizes the latent vector in the $W^+$ space to minimize both pixel-wise MSE loss and VGG perceptual similarity loss:

```bash
python3 invert.py --target /path/to/target_face.png --steps 1000 --lr 0.01
```

### Inversion Argument Reference:
* `--target`: Path to the target image you want to project into the latent space (required).
* `--weights`: Path to pre-trained weights (default: `weights/best_checkpoint_1010k.pth`).
* `--output_dir`: Directory to save the output files (default: `output`).
* `--steps`: Number of optimization steps (default: 1000).
* `--lr`: Optimizer learning rate (default: 0.01).
* `--device`: Force device selection (`cuda`, `mps`, `cpu`).

---

## License

This project is licensed under the MIT License.
