import os
import argparse
import random
import numpy as np
import torch
import torchvision.utils as vutils
from models import Generator

def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def estimate_w_avg(generator, device, latent_dim=512, num_samples=10000, batch_size=100):
    """
    Estimates w_avg (average w latent vector) for the truncation trick.
    """
    print(f"Estimating w_avg using {num_samples} samples...")
    w_accum = []
    generator.eval()
    for _ in range(num_samples // batch_size):
        z = torch.randn(batch_size, latent_dim, device=device)
        w = generator.g_mapping(z)
        w_accum.append(w.mean(dim=0, keepdim=True))
    w_avg = torch.stack(w_accum).mean(dim=0)
    return w_avg

def main():
    parser = argparse.ArgumentParser(description="Generate images using pre-trained StyleGAN2 model")
    parser.add_argument("--weights", type=str, default="weights/best_checkpoint_1010k.pth", help="Path to checkpoint weights")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save generated images")
    parser.add_argument("--num_images", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--truncation", type=float, default=0.7, help="Truncation factor for truncation trick (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--grid", type=bool, default=True, help="Whether to save images as a single grid (else saves individually)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda, mps, cpu)")

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Set random seed
    set_seed(args.seed)

    # Initialize generator
    generator = Generator().to(device)

    # Load weights
    if not os.path.exists(args.weights):
        if args.weights == "weights/best_checkpoint_1010k.pth":
            os.makedirs(os.path.dirname(args.weights), exist_ok=True)
            url = "https://huggingface.co/pixelnotfound/stylegan2/resolve/main/best_checkpoint_1010k.pth"
            print(f"Weights file not found at '{args.weights}'. Downloading pre-trained weights from Hugging Face...")
            import urllib.request
            
            def progress_bar(block_num, block_size, total_size):
                read_so_far = block_num * block_size
                if total_size > 0:
                    percent = min(100, read_so_far * 100 // total_size)
                    import sys
                    sys.stdout.write(f"\rDownloading: {percent}% ({read_so_far // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
                    sys.stdout.flush()
            
            urllib.request.urlretrieve(url, args.weights, reporthook=progress_bar)
            print("\nDownload complete!")
        else:
            raise FileNotFoundError(f"Weights file not found at: {args.weights}")
    
    print(f"Loading weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device)
    
    # Standard checkpoints can have g_running_state_dict (EMA model) or g_state_dict
    if 'g_running_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['g_running_state_dict'])
        print("Successfully loaded EMA generator (g_running_state_dict)")
    elif 'g_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['g_state_dict'])
        print("Successfully loaded generator (g_state_dict)")
    else:
        # Direct state dict
        generator.load_state_dict(checkpoint)
        print("Loaded weights directly as state_dict")

    generator.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Estimate w_avg for truncation trick if requested
    w_avg = None
    if args.truncation < 1.0:
        w_avg = estimate_w_avg(generator, device, latent_dim=512)

    # Generate images
    print(f"Generating {args.num_images} images with truncation {args.truncation}...")
    z = torch.randn(args.num_images, 512, device=device)
    
    images = generator(z, truncation=args.truncation, w_avg=w_avg)
    
    # Post-process: convert [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    if args.grid:
        grid_path = os.path.join(args.output_dir, "grid.png")
        nrow = int(np.sqrt(args.num_images))
        vutils.save_image(images, grid_path, nrow=nrow, padding=2)
        print(f"Saved image grid to: {grid_path}")
    else:
        for idx in range(args.num_images):
            img_path = os.path.join(args.output_dir, f"sample_{idx:04d}.png")
            vutils.save_image(images[idx], img_path)
            print(f"Saved image to: {img_path}")

    print("Generation complete!")

if __name__ == "__main__":
    main()
