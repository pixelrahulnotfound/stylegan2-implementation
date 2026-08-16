import os
import math
import time
import random
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# Use torchmetrics for FID if available, otherwise define a placeholder or handle import gracefully
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

from models import Generator, Discriminator
from losses import d_loss, g_loss, r1_loss, g_path_regularize, EMA, requires_grad

def get_dataloader(data_path, image_size, batch_size=8, num_workers=1):
    """
    Creates and returns a dataloader for the dataset.
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(root=data_path, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return dataloader

def get_params_with_lr(model):
    """
    Separates the parameters of the mapping network from the rest of the generator.
    """
    mapping_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'g_mapping' in name:
            mapping_params.append(param)
        else:
            other_params.append(param)
    return mapping_params, other_params

def resize_images(images, size=(299, 299)):
    """
    Resizes images for the InceptionV3 model used by FID.
    """
    transform = transforms.Compose([
        transforms.Resize(size),
    ])
    return transform(images)

def add_fake_images(fid_metric, g_running, num_images, batch_size, latent_dim, device):
    """
    Generates fake images and updates the FID metric.
    """
    g_running.eval()
    with torch.no_grad():
        for _ in tqdm(range(0, num_images, batch_size), desc="Generating fake images for FID"):
            z = torch.randn(batch_size, latent_dim, device=device)
            batch_images = g_running(z)
            torch.cuda.synchronize(device)
        
            # Prepare images for InceptionV3
            resize_batch = resize_images(batch_images)
            resize_batch = ((resize_batch + 1) * 127.5).clamp(0, 255)
            resize_batch = resize_batch.to(torch.uint8)
            
            fid_metric.update(resize_batch, real=False)
            torch.cuda.empty_cache()

def add_real_images(fid_metric, dataloader, num_images, batch_size, device):
    """
    Loads real images and updates the FID metric.
    """
    count = 0
    for batch in tqdm(dataloader, desc="Processing real images for FID"):
        imgs, _ = batch
        imgs = resize_images(imgs)
        imgs = imgs.to(device)
        imgs = ((imgs + 1) * 127.5).clamp(0, 255)
        imgs = imgs.to(torch.uint8)
        
        fid_metric.update(imgs, real=True)
        count += imgs.size(0)
        
        torch.cuda.empty_cache()
        if count >= num_images:
            break

def calculate_and_save_fid(fid_metric, iteration, dataloader, g_running, num_fake_images, batch_size, latent_dim, device, fid_file):
    """
    Calculates FID score, logs it, and saves to file.
    """
    if not HAS_TORCHMETRICS:
        print("torchmetrics not installed. Skipping FID calculation.")
        return
        
    fid_metric.reset()
    add_fake_images(fid_metric, g_running, num_fake_images, batch_size, latent_dim, device)
    add_real_images(fid_metric, dataloader, num_fake_images, batch_size, device)
    
    fid_score = fid_metric.compute()
    print(f"\nFID score at iteration {iteration}: {fid_score.item()}")
    
    with open(fid_file, 'a') as f:
        f.write(f"Iteration {iteration}: {fid_score.item()}\n")

def main():
    parser = argparse.ArgumentParser(description="StyleGAN2 Training Script")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to image dataset directory")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=str, default="16", help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0025, help="Base learning rate")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of latent space")
    parser.add_argument("--total_iters", type=int, default=1000000, help="Total training iterations")
    parser.add_argument("--eval_freq", type=int, default=10000, help="FID evaluation and checkpoint saving frequency")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader")
    
    args = parser.parse_args()
    batch_size = int(args.batch_size)

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Optimization setting to prevent memory fragmentation on GPU
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Set up run directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.checkpoints_dir, f"run_{timestamp}")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    sample_dir = os.path.join(run_dir, "samples")
    fid_file = os.path.join(run_dir, 'fid.txt')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Initialize networks
    g = Generator().to(device)
    d = Discriminator().to(device)
    g_running = Generator().to(device)
    g_running.train(False)
    
    # Initialize EMA weights
    EMA(g_running, g, decay=0.0)

    # Configure learning rates and optimizer parameter groups
    mapping_params, other_params = get_params_with_lr(g)
    mapping_lr = args.lr * 0.01

    g_reg_freq = 4
    d_reg_freq = 16

    g_reg_adjustment = g_reg_freq / (g_reg_freq + 1)
    d_reg_adjustment = d_reg_freq / (d_reg_freq + 1)

    g_optimizer = torch.optim.Adam([
        {'params': mapping_params, 'lr': mapping_lr},
        {'params': other_params, 'lr': args.lr * g_reg_adjustment}
    ], betas=(0.0 ** g_reg_adjustment, 0.99 ** g_reg_adjustment))
    
    d_optimizer = torch.optim.Adam(
        d.parameters(),
        lr=args.lr * d_reg_adjustment,
        betas=(0.0 ** d_reg_adjustment, 0.99 ** d_reg_adjustment)
    )

    start_iter = 0
    mean_path_length = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_iter = checkpoint['iteration']
            g.load_state_dict(checkpoint['g_state_dict'])
            d.load_state_dict(checkpoint['d_state_dict'])
            g_running.load_state_dict(checkpoint['g_running_state_dict'])
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            mean_path_length = checkpoint.get('mean_path_length', 0.0)
            print(f"=> Loaded checkpoint '{args.resume}' (iteration {start_iter})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")

    # Initialize FID metric
    fid_metric = None
    if HAS_TORCHMETRICS:
        fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    # Dataloader
    try:
        dataloader = get_dataloader(args.data_path, 256, batch_size, args.num_workers)
        dataset_iter = iter(dataloader)
        print(f"Dataset successfully loaded from: {args.data_path}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("Please check your data_path argument and structure.")
        return

    print(f"Starting training at iteration {start_iter}...")
    pbar = tqdm(range(start_iter, args.total_iters))
    
    for i in pbar:
        # ------------------ Train Discriminator ------------------
        requires_grad(g, False)
        requires_grad(d, True)

        try:
            real_imgs, _ = next(dataset_iter)
        except StopIteration:
            dataset_iter = iter(dataloader)
            real_imgs, _ = next(dataset_iter)

        real_imgs = real_imgs.to(device)
        real_preds = d(real_imgs)

        # Generate fake images
        z = torch.randn(real_imgs.size(0), args.latent_dim, device=device)
        fake_imgs = g(z)
        fake_preds = d(fake_imgs.detach())

        d_loss_val = d_loss(real_preds, fake_preds)

        d_optimizer.zero_grad()
        d_loss_val.backward()
        d_optimizer.step()

        # Lazy regularization for Discriminator: R1 loss
        if i % d_reg_freq == 0:
            real_imgs.requires_grad_(True)
            real_preds = d(real_imgs)
            r1_loss_val = r1_loss(real_preds, real_imgs)
            
            d_optimizer.zero_grad()
            gamma = 0.8192
            r1_reg = ((gamma * 0.5) * r1_loss_val * d_reg_freq + 0.0 * real_preds[0])
            r1_reg.backward()
            d_optimizer.step()
            
            torch.cuda.empty_cache()

        # ------------------ Train Generator ------------------
        requires_grad(g, True)
        requires_grad(d, False)

        z = torch.randn(real_imgs.size(0), args.latent_dim, device=device)
        fake_imgs = g(z)
        fake_preds = d(fake_imgs)

        g_loss_val = g_loss(fake_preds)

        g_optimizer.zero_grad()
        g_loss_val.backward()
        g_optimizer.step()

        # Lazy regularization for Generator: Path Length loss
        if i % g_reg_freq == 0:
            z = torch.randn(real_imgs.size(0), args.latent_dim, device=device)
            fake_imgs, latents = g(z, return_latents=True)
            
            ppl_loss, mean_path_length, _ = g_path_regularize(
                fake_imgs, latents, mean_path_length
            )
            
            g_optimizer.zero_grad()
            ppl_loss = 2.0 * g_reg_freq * ppl_loss
            ppl_loss.backward()
            g_optimizer.step()
            
            torch.cuda.empty_cache()

        # Update EMA model
        EMA(g_running, g, decay=0.999)

        # Progress bar description
        pbar.set_description(
            f"D Loss: {d_loss_val.item():.4f} | G Loss: {g_loss_val.item():.4f}"
        )

        # Save sample images and evaluate FID
        if i > 0 and i % args.eval_freq == 0:
            # Generate sample grid with EMA model
            g_running.eval()
            with torch.no_grad():
                sample_z = torch.randn(16, args.latent_dim, device=device)
                sample_imgs = g_running(sample_z)
                save_image(
                    sample_imgs,
                    f'{sample_dir}/sample_iter_{i}.png',
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1)
                )
            
            # Save checkpoint
            torch.save({
                'g_state_dict': g.state_dict(),
                'g_running_state_dict': g_running.state_dict(),
                'd_state_dict': d.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'iteration': i,
                'mean_path_length': mean_path_length,
            }, f'{checkpoint_dir}/checkpoint_iter_{i}.pth')

            # Calculate FID with error handling (to avoid crashes from memory leaks or segfaults)
            if HAS_TORCHMETRICS:
                try:
                    torch.cuda.synchronize(device)
                    calculate_and_save_fid(
                        fid_metric, i, dataloader, g_running,
                        num_fake_images=5000, batch_size=batch_size,
                        latent_dim=args.latent_dim, device=device, fid_file=fid_file
                    )
                except Exception as e:
                    print(f"\nFID calculation failed at iteration {i}: {str(e)}")
                    torch.cuda.empty_cache()

    # Save final model
    torch.save({
        'g_state_dict': g.state_dict(),
        'g_running_state_dict': g_running.state_dict(),
        'd_state_dict': d.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'iteration': args.total_iters,
        'mean_path_length': mean_path_length,
    }, f'{run_dir}/completed_checkpoint.pth')
    print("Training finished. Final checkpoint saved.")

if __name__ == "__main__":
    main()
