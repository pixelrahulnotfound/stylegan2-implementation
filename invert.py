import os
import argparse
import math
import types
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import cv2
from models import Generator

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss based on intermediate features of a pre-trained VGG16 network.
    Uses standard ImageNet normalization for correct perceptual gradient computation.
    """
    def __init__(self, device):
        super().__init__()
        # Load pre-trained VGG16
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
        
        # We slice VGG16 to extract features from different layers (relu1_2, relu2_2, relu3_3, relu4_3)
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16])
        self.slice4 = nn.Sequential(*list(vgg.children())[16:23])
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Scale input from [-1, 1] to [0, 1]
        x = (x + 1) / 2.0
        
        # Normalize with ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]

def get_perceptual_loss(features_gen, features_target):
    loss = 0.0
    weights = [1.0, 1.0, 1.0, 1.0]
    for f_gen, f_target, w in zip(features_gen, features_target, weights):
        loss += w * torch.mean((f_gen - f_target) ** 2)
    return loss

def crop_and_align_face(image_path, output_path=None, size=256, padding_ratio=2.0):
    """
    Detects the main face in the image using OpenCV Haar Cascades with multiple fallbacks,
    crops it with padding (FFHQ style), and resizes it to the target size.
    If no face is detected, returns the original image center-cropped.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found at: {image_path}")
        
    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil)
    
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Try multiple cascades for robustness (alt2 is best for faces with sunglasses/tilts)
    cascades = [
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_frontalface_alt.xml',
        'haarcascade_frontalface_default.xml',
        'haarcascade_profileface.xml'
    ]
    
    faces = []
    for cascade_name in cascades:
        cascade_path = os.path.join(cv2.data.haarcascades, cascade_name)
        if not os.path.exists(cascade_path):
            continue
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Try parameter settings from conservative to sensitive
        for sf in [1.05, 1.1, 1.2]:
            for mn in [3, 4, 5]:
                detected = face_cascade.detectMultiScale(
                    gray, scaleFactor=sf, minNeighbors=mn, minSize=(30, 30)
                )
                if len(detected) > 0:
                    faces = detected
                    print(f"Face successfully detected using {cascade_name} (scaleFactor={sf}, minNeighbors={mn})")
                    break
            if len(faces) > 0:
                break
        if len(faces) > 0:
            break
            
    if len(faces) == 0:
        print("No faces detected in the image. Defaulting to center crop...")
        w_img, h_img = img_pil.size
        crop_size = min(w_img, h_img)
        left = (w_img - crop_size) // 2
        top = (h_img - crop_size) // 2
        cropped_img = img_pil.crop((left, top, left+crop_size, top+crop_size))
        cropped_img = cropped_img.resize((size, size), Image.Resampling.LANCZOS)
        if output_path:
            cropped_img.save(output_path)
        return cropped_img

    # Select the largest face detected
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    
    # Calculate face center
    cx = x + w / 2.0
    cy = y + h / 2.0
    
    # Crop size with padding (FFHQ style)
    crop_size = int(max(w, h) * padding_ratio)
    
    # Crop bounds
    x1 = int(cx - crop_size / 2.0)
    y1 = int(cy - crop_size / 2.0)
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    
    h_img, w_img, _ = img_np.shape
    
    # Handle coordinates going out of image bounds by edge padding
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w_img)
    pad_bottom = max(0, y2 - h_img)
    
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded_np = np.pad(
            img_np,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='edge'
        )
        x1_p = x1 + pad_left
        y1_p = y1 + pad_top
        x2_p = x2 + pad_left
        y2_p = y2 + pad_top
        crop = padded_np[y1_p:y2_p, x1_p:x2_p]
    else:
        crop = img_np[y1:y2, x1:x2]
        
    cropped_img = Image.fromarray(crop)
    cropped_img = cropped_img.resize((size, size), Image.Resampling.LANCZOS)
    
    if output_path:
        cropped_img.save(output_path)
        print(f"Saved cropped/aligned face to: {output_path}")
        
    return cropped_img

def load_and_preprocess_image(image_path, size=256):
    """
    Loads an image and preprocesses it to shape (1, 3, size, size) and range [-1, 1].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found at: {image_path}")
        
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img).unsqueeze(0)

@torch.no_grad()
def get_w_avg(generator, device, latent_dim=512, num_samples=10000, batch_size=100):
    """
    Computes average w latent code to initialize the projection optimization.
    """
    print("Estimating w_avg for latent space initialization...")
    w_accum = []
    generator.eval()
    for _ in range(num_samples // batch_size):
        z = torch.randn(batch_size, latent_dim, device=device)
        w = generator.g_mapping(z)
        w_accum.append(w.mean(dim=0, keepdim=True))
    return torch.stack(w_accum).mean(dim=0)

def save_reconstruction(tensor_img, fp):
    # tensor_img: (1, 3, H, W) in range [-1, 1]
    img = (tensor_img.squeeze(0) + 1) / 2
    img = torch.clamp(img, 0, 1)
    img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
    img_uint8 = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(fp)

def synthesis_forward(generator, styles):
    """
    Manually perform Generator synthesis pass with a list of style vectors.
    styles must have shape (batch_size, 7, 512).
    """
    out = generator.block_4x4(styles[:, 0])
    out_4 = generator.to_rgb_4(out)
    out_4 = generator.upsample(out_4)
    
    out = generator.block_8x8(styles[:, 1], out)
    out_8 = generator.to_rgb_8(out)
    out_8 += out_4 * (1 / math.sqrt(2))
    out_8 = generator.upsample(out_8)
    
    out = generator.block_16x16(styles[:, 2], out)
    out_16 = generator.to_rgb_16(out)
    out_16 += out_8 * (1 / math.sqrt(2))
    out_16 = generator.upsample(out_16)
    
    out = generator.block_32x32(styles[:, 3], out)
    out_32 = generator.to_rgb_32(out)
    out_32 += out_16 * (1 / math.sqrt(2))
    out_32 = generator.upsample(out_32)
    
    out = generator.block_64x64(styles[:, 4], out)
    out_64 = generator.to_rgb_64(out)
    out_64 += out_32 * (1 / math.sqrt(2))
    out_64 = generator.upsample(out_64)

    out = generator.block_128x128(styles[:, 5], out)            
    out_128 = generator.to_rgb_128(out)
    out_128 += out_64 * (1 / math.sqrt(2))
    out_128 = generator.upsample(out_128)

    out = generator.block_256x256(styles[:, 6], out)
    out_256 = generator.to_rgb_128(out)  # Match the trained model routing
    out_256 += out_128 * (1 / math.sqrt(2))
    return out_256

def main():
    parser = argparse.ArgumentParser(description="Perform GAN Inversion (Project a face photo into the StyleGAN2 latent space)")
    parser.add_argument("--target", type=str, required=True, help="Path to target image to invert")
    parser.add_argument("--weights", type=str, default="weights/best_checkpoint_1010k.pth", help="Path to checkpoint weights")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output reconstruction and latent code")
    parser.add_argument("--steps", type=int, default=1000, help="Number of optimization steps")
    parser.add_argument("--lr", type=float, default=0.05, help="Optimization learning rate (default: 0.05)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda, mps, cpu)")
    parser.add_argument("--no_align", action="store_true", help="Disable automatic face detection and alignment")
    parser.add_argument("--optimize_noise", action="store_true", help="Optimize noise tensors as well as latent codes")
    parser.add_argument("--reg_w", type=float, default=1.0, help="Weight for W+ latent variance regularization (default: 1.0)")
    parser.add_argument("--padding_ratio", type=float, default=2.0, help="Face crop padding ratio (default: 2.0)")

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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Face alignment pre-processing
    if not args.no_align:
        aligned_path = os.path.join(args.output_dir, "inversion_target_cropped.png")
        print(f"Aligning and cropping face from: {args.target}")
        crop_and_align_face(args.target, aligned_path, size=256, padding_ratio=args.padding_ratio)
        target_img_path = aligned_path
    else:
        print("Automatic face alignment disabled. Using target image directly.")
        target_img_path = args.target

    # Load target image
    target_img = load_and_preprocess_image(target_img_path).to(device)
    save_reconstruction(target_img, os.path.join(args.output_dir, "inversion_target.png"))

    # 2. Initialize generator
    generator = Generator().to(device)
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
            
    checkpoint = torch.load(args.weights, map_location=device)
    if 'g_running_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['g_running_state_dict'])
        print("Loaded EMA generator.")
    elif 'g_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['g_state_dict'])
        print("Loaded generator.")
    else:
        generator.load_state_dict(checkpoint)
        print("Loaded state_dict directly.")
    generator.eval()

    # 2.1 Patch NoiseLayer to use static noise during inversion
    noise_layers = []
    for name, module in generator.named_modules():
        if module.__class__.__name__ == 'NoiseLayer':
            noise_layers.append(module)
            
    def patched_noise_forward(self, x, noise=None):
        if noise is not None:
            return x + (noise * self.weight)
        if hasattr(self, 'static_noise') and self.static_noise is not None:
            return x + (self.static_noise.to(x.device) * self.weight)
        # Fallback to random noise
        N, _, H, W = x.shape
        noise = torch.randn(N, 1, H, W, device=x.device)
        return x + (noise * self.weight)
        
    def capture_forward(self, x, noise=None):
        N, _, H, W = x.shape
        # Create standard normal noise
        self.static_noise = torch.randn(N, 1, H, W, device=x.device)
        return x + (self.static_noise * self.weight)
        
    # Apply capture forward temporarily
    for module in noise_layers:
        module.forward = types.MethodType(capture_forward, module)
        
    # Run a single dummy pass to capture noise shapes
    with torch.no_grad():
        dummy_z = torch.randn(1, 512, device=device)
        _ = generator(dummy_z)
        
    # Re-apply the permanent patched forward
    for module in noise_layers:
        module.forward = types.MethodType(patched_noise_forward, module)

    # 3. Compute initialization (w_avg)
    w_avg = get_w_avg(generator, device)
    
    # We optimize in the W+ space: shape (1, 7, 512) for the 7 layers of styles
    # We initialize it with w_avg replicated for all 7 layers
    w_opt = w_avg.unsqueeze(0).repeat(1, 7, 1).clone().detach()
    w_opt.requires_grad = True

    # Configure noise parameters for optimization if requested
    if args.optimize_noise:
        for module in noise_layers:
            module.static_noise.requires_grad = True

    # 4. Initialize optimizer, scheduler, and loss functions
    optimizer_params = [w_opt]
    if args.optimize_noise:
        for module in noise_layers:
            optimizer_params.append(module.static_noise)
            
    optimizer = optim.Adam(optimizer_params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.01)
    
    mse_loss_fn = nn.MSELoss()
    
    # Load perceptual loss model
    try:
        perceptual_loss_fn = VGGPerceptualLoss(device)
        target_features = perceptual_loss_fn(target_img)
        print("VGG Perceptual Loss initialized successfully.")
    except Exception as e:
        perceptual_loss_fn = None
        print(f"Failed to load VGG Perceptual Loss ({str(e)}). Falling back to MSE/L1-only optimization.")

    # 5. Optimization Loop
    print(f"Starting optimization for {args.steps} steps...")
    for step in range(args.steps):
        # Forward pass using synthesis helper
        generated_img = synthesis_forward(generator, w_opt)

        # Calculate pixel-level losses
        loss_mse = mse_loss_fn(generated_img, target_img)
        loss_l1 = torch.mean(torch.abs(generated_img - target_img))
        
        # Calculate perceptual loss
        if perceptual_loss_fn is not None:
            gen_features = perceptual_loss_fn(generated_img)
            loss_perceptual = get_perceptual_loss(gen_features, target_features)
            loss = 1.0 * loss_l1 + 1.0 * loss_mse + 1.0 * loss_perceptual
        else:
            loss = 1.0 * loss_l1 + 1.0 * loss_mse
            
        # Regularization on W+ latent space (forces styles across layers to remain similar)
        if args.reg_w > 0:
            w_mean = w_opt.mean(dim=1, keepdim=True)
            loss_w_reg = torch.mean((w_opt - w_mean) ** 2)
            loss += args.reg_w * loss_w_reg

        # Regularization on noise parameters to keep them normal
        if args.optimize_noise:
            loss_noise = 0.0
            for module in noise_layers:
                n = module.static_noise
                loss_noise += torch.mean(n) ** 2 + (torch.std(n) - 1.0) ** 2
            loss += loss_noise * 1e5

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Print progress and save intermediate results
        if step % 100 == 0 or step == args.steps - 1:
            print(f"Step {step}/{args.steps} | Loss: {loss.item():.6f} (MSE: {loss_mse.item():.6f}, L1: {loss_l1.item():.6f})")
            save_reconstruction(generated_img, os.path.join(args.output_dir, "inversion_output.png"))

    # Save final results
    latent_save_path = os.path.join(args.output_dir, "inversion_latent.pt")
    torch.save(w_opt.detach().cpu(), latent_save_path)
    print(f"\nOptimization complete!")
    print(f"Saved reconstructed image to: {os.path.join(args.output_dir, 'inversion_output.png')}")
    print(f"Saved optimized latent code to: {latent_save_path}")

    # 6. Generate face variations based on the inverted face
    print("\nGenerating variations based on your inverted face...")
    with torch.no_grad():
        z_random = torch.randn(1, 512, device=device)
        w_random = generator.g_mapping(z_random).unsqueeze(1).repeat(1, 7, 1)
        
        for alpha in [0.1, 0.3, 0.5, 0.7]:
            w_mixed = w_opt + alpha * (w_random - w_opt)
            
            # Generate style variation
            out_256 = synthesis_forward(generator, w_mixed)
            
            blend_path = os.path.join(args.output_dir, f"inversion_blend_{alpha:.1f}.png")
            save_reconstruction(out_256, blend_path)
            print(f"Saved blend variation ({alpha*100:.0f}% random features) to: {blend_path}")

if __name__ == "__main__":
    main()
