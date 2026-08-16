import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EqualLRConv2d(nn.Module):
    """
    Convolutional layer with Equalized Learning Rate.
    Normalizes the weights at runtime to stabilize training.
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * (kernel_size ** 2))
        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, "
            f"{self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )

class EqualLRLinear(nn.Module):
    """
    Linear layer with Equalized Learning Rate.
    """
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0.0, lr_mul=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        bias = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, self.weight * self.scale, bias=bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"

class PixelNorm(nn.Module):
    """
    Pixelwise feature vector normalization.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class MiniBatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer to improve discriminator variety.
    """
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size
    
    def forward(self, x):
        N, C, H, W = x.shape
        G = min(self.group_size, N)
        
        # Reshape to group size
        y = x.view(G, -1, C, H, W)
        # Subtract mean over group
        y = y - torch.mean(y, dim=0, keepdim=True)
        # Variance
        y = torch.mean(torch.square(y), dim=0)
        # Std dev
        y = torch.sqrt(y + 1e-8)
        # Average over all channels and spatial dims
        y = torch.mean(y, dim=[1, 2, 3], keepdim=True)
        # Replicate and concat
        y = y.repeat(G, 1, H, W)
        return torch.cat([x, y], dim=1)

class LearnedConstant(nn.Module):
    """
    Learned constant input layer for the generator start block.
    """
    def __init__(self, in_c):
        super().__init__()
        self.constant = nn.Parameter(torch.randn(1, in_c, 4, 4))
        
    def forward(self, batch_size):
        return self.constant.expand(batch_size, -1, -1, -1)

class NoiseLayer(nn.Module):
    """
    Noise injection layer.
    """
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x, noise=None):
        if noise is None:
            N, _, H, W = x.shape
            noise = torch.randn(N, 1, H, W, device=x.device)
        return x + (noise * self.weight)

class MappingNetwork(nn.Module):
    """
    Mapping network to map latent z to w space.
    """
    def __init__(self):
        super().__init__()
        self.norm = PixelNorm()
        self.layers = nn.Sequential(
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLRLinear(512, 512),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        x = self.norm(x)
        return self.layers(x)

class Blur(nn.Module):
    """
    Anti-aliasing blur layer.
    """
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 3, 3, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        f = f / f.sum()
        return F.conv2d(x, f.expand(x.size(1), -1, -1, -1), groups=x.size(1), padding=1)

class Conv2d_mod(nn.Module):
    """
    Weight modulated and demodulated convolution layer.
    Replaces AdaIN in StyleGAN2.
    """
    def __init__(self, in_channels, out_channels, kernel_size, latent_dim, padding):
        super().__init__()
        # 1 x out_c x in_c x k x k
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channels * (kernel_size ** 2))
        
        # Style modulation layer
        self.modulation = EqualLRLinear(latent_dim, in_channels, bias_init=1.0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x, style):
        batch, channels, height, width = x.shape
        
        # Modulation: scale = A(w)
        style = self.modulation(style).view(batch, 1, -1, 1, 1)
        
        # Modulate weights
        weight = self.scale * self.weight * style
        
        # Demodulation: normalize the output features standard deviation
        demod = torch.rsqrt(weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)
        
        # Group conv implementation: treat batch as groups
        weight = weight.view(
            batch * self.out_channels, self.in_channels,
            self.kernel_size, self.kernel_size
        )
        
        x = x.view(1, batch * channels, height, width)
        out = F.conv2d(
            x,
            weight,
            padding=self.padding,
            groups=batch
        )
        
        _, _, out_h, out_w = out.shape
        out = out.view(batch, self.out_channels, out_h, out_w)
        return out

class g_style_block(nn.Module):
    """
    Style block in the Generator containing modulation, noise injection, and activation.
    """
    def __init__(self, in_c, out_c, ksize, padding, upsample=True, latent_dim=512):
        super().__init__()
        layers_list = []

        if upsample:
            layers_list.extend([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                Conv2d_mod(in_c, out_c, ksize, latent_dim, padding=padding),
                NoiseLayer(out_c),
            ])
        else:
            self.learned_constant = LearnedConstant(in_c)

        layers_list.extend([
            nn.LeakyReLU(0.2),
            Conv2d_mod(out_c, out_c, ksize, latent_dim, padding=padding),
            NoiseLayer(out_c),
            nn.LeakyReLU(0.2),
        ])

        self.layers = nn.ModuleList(layers_list)
        self.upsample = upsample
        
    def forward(self, w, x=None):
        if not self.upsample:
            x = self.learned_constant(w.size(0))
        
        for layer in self.layers:
            if isinstance(layer, Conv2d_mod):
                x = layer(x, w)
            else:
                x = layer(x)
            
        return x

class Generator(nn.Module):
    """
    Generator network using skip connections to progressively output images at the final resolution.
    """
    def __init__(self, in_c=512):
        super().__init__()
        self.g_mapping = MappingNetwork()

        fmaps = 0.5
        in_c = int(in_c * fmaps)
        
        self.block_4x4 = g_style_block(in_c, in_c, 3, 1, upsample=False)
        self.block_8x8 = g_style_block(in_c, in_c, 3, 1)
        self.block_16x16 = g_style_block(in_c, in_c, 3, 1)
        self.block_32x32 = g_style_block(in_c, in_c, 3, 1)
        self.block_64x64 = g_style_block(in_c, in_c//2, 3, 1)
        self.block_128x128 = g_style_block(in_c//2, in_c//4, 3, 1)
        self.block_256x256 = g_style_block(in_c//4, in_c//4, 3, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.to_rgb_4 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_8 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_16 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_32 = EqualLRConv2d(in_c, 3, 1)
        self.to_rgb_64 = EqualLRConv2d(in_c//2, 3, 1)
        self.to_rgb_128 = EqualLRConv2d(in_c//4, 3, 1)
        self.to_rgb_256 = EqualLRConv2d(in_c//4, 3, 1)

    def forward(self, z, return_latents=False, truncation=1.0, w_avg=None):
        w = self.g_mapping(z)
        
        if w_avg is not None and truncation < 1.0:
            w = w_avg + truncation * (w - w_avg)
            
        batch_size = z.size(0)
        
        # Style mixing regularization during training
        mixing = (torch.rand(batch_size, device=z.device) < 0.9) if (self.training and w_avg is None) else False
        
        if mixing:
            z2 = torch.randn_like(z)
            w2 = self.g_mapping(z2)
            crossover_points = torch.randint(1, 7, (batch_size,), device=z.device)
            
            styles = []
            for layer_idx in range(7):
                use_w2 = crossover_points <= layer_idx
                style = torch.where(use_w2.unsqueeze(1), w2, w)
                styles.append(style)
        else:
            styles = [w] * 7
        
        # Progressive generation blocks with skip-connection summation
        out = self.block_4x4(styles[0])
        out_4 = self.to_rgb_4(out)
        out_4 = self.upsample(out_4)
        
        out = self.block_8x8(styles[1], out)
        out_8 = self.to_rgb_8(out)
        out_8 += out_4 * (1 / math.sqrt(2))
        out_8 = self.upsample(out_8)
        
        out = self.block_16x16(styles[2], out)
        out_16 = self.to_rgb_16(out)
        out_16 += out_8 * (1 / math.sqrt(2))
        out_16 = self.upsample(out_16)
        
        out = self.block_32x32(styles[3], out)
        out_32 = self.to_rgb_32(out)
        out_32 += out_16 * (1 / math.sqrt(2))
        out_32 = self.upsample(out_32)
        
        out = self.block_64x64(styles[4], out)
        out_64 = self.to_rgb_64(out)
        out_64 += out_32 * (1 / math.sqrt(2))
        out_64 = self.upsample(out_64)

        out = self.block_128x128(styles[5], out)            
        out_128 = self.to_rgb_128(out)
        out_128 += out_64 * (1 / math.sqrt(2))
        out_128 = self.upsample(out_128)

        out = self.block_256x256(styles[6], out)
        out_256 = self.to_rgb_128(out)
        out_256 += out_128 * (1 / math.sqrt(2))
        
        if return_latents:
            return out_256, styles[6]
        else:
            return out_256

class d_style_block(nn.Module):
    """
    Discriminator block containing residual connections and downsampling.
    """
    def __init__(self, in_c, out_c, ksize1, padding, ksize2=None, padding2=None, mbatch=False):
        super().__init__()

        if ksize2 is None:
            ksize2 = ksize1
        if padding2 is None:
            padding2 = padding
        
        self.down_res = nn.Sequential(
            Blur(),
            EqualLRConv2d(in_c, out_c, 3, padding=1, stride=2),
            EqualLRConv2d(out_c, out_c, 2, padding=0, stride=1) if mbatch else nn.Identity()
        )
        
        layers_list = []
        if mbatch:
            layers_list.append(MiniBatchStdDev())
            in_c += 1
            
        layers_list.extend([
            EqualLRConv2d(in_c, in_c, ksize1, padding=padding),
            nn.LeakyReLU(0.2),
            EqualLRConv2d(in_c, out_c, ksize2, padding=padding2),
            nn.LeakyReLU(0.2),
        ])

        if not mbatch:
            layers_list.append(EqualLRConv2d(out_c, out_c, 3, padding=1, stride=2))
        
        self.layers = nn.ModuleList(layers_list)
    
    def forward(self, x):
        res = self.down_res(x)
        for layer in self.layers:
            x = layer(x)
        return (res + x) * (1 / math.sqrt(2))

class Discriminator(nn.Module):
    """
    Discriminator network built with residual connections.
    """
    def __init__(self, out_c=512):
        super().__init__()

        fmaps = 0.5
        out_c = int(out_c * fmaps)

        self.from_rgb = EqualLRConv2d(3, out_c//4, 1)

        self.block_256x256 = d_style_block(out_c//4, out_c//4, 3, 1)
        self.block_128x128 = d_style_block(out_c//4, out_c//2, 3, 1)
        self.block_64x64 = d_style_block(out_c//2, out_c, 3, 1)
        self.block_32x32 = d_style_block(out_c, out_c, 3, 1)
        self.block_16x16 = d_style_block(out_c, out_c, 3, 1)
        self.block_8x8 = d_style_block(out_c, out_c, 3, 1)
        self.block_4x4 = d_style_block(out_c, out_c, 3, 1, 4, 0, mbatch=True)

        self.linear = EqualLRLinear(out_c, 1)

    def forward(self, x):
        out = self.from_rgb(x)
        out = self.block_256x256(out)
        out = self.block_128x128(out)
        out = self.block_64x64(out)
        out = self.block_32x32(out)
        out = self.block_16x16(out)
        out = self.block_8x8(out)
        out = self.block_4x4(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
