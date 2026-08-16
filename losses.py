import math
import torch
import torch.nn.functional as F

def d_loss(real_pred, fake_pred):
    """
    Standard logistic GAN loss for the discriminator.
    """
    real_loss = F.softplus(-real_pred).mean()
    fake_loss = F.softplus(fake_pred).mean()
    return real_loss + fake_loss

def g_loss(fake_pred):
    """
    Standard logistic GAN loss for the generator.
    """
    return F.softplus(-fake_pred).mean()

def r1_loss(real_pred, real_images):
    """
    R1 regularization penalty.
    Computes gradients of discriminator predictions with respect to real images.
    """
    with torch.set_grad_enabled(True):
        grad_real, = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_images,
            grad_outputs=torch.ones([], device=real_pred.device),
            create_graph=True,
            only_inputs=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    """
    Perceptual Path Length (PPL) regularization for the generator.
    Encourages a smoother mapping from latent space to image space.
    """
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    
    grad, = torch.autograd.grad(
        outputs=(fake_img * noise).sum(),
        inputs=latents,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )
    
    path_lengths = torch.sqrt(grad.pow(2).sum(1).mean(0))
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = (path_lengths - path_mean).pow(2).mean()

    del noise, grad
    torch.cuda.empty_cache()
    
    return path_penalty, path_mean.detach(), path_lengths

def EMA(model1, model2, decay=0.999):
    """
    Exponential Moving Average of model parameters.
    Accumulates weights of model2 into model1.
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1.0 - decay)

def requires_grad(model, flag=True):
    """
    Utility function to toggle gradient computation for model parameters.
    """
    for p in model.parameters():
        p.requires_grad = flag
