import torch
import torch.nn.functional as F


def masked_mean(v, mask, eps=1e-6):
    num = (v * mask).sum(dim=(1, 2, 3))
    den = (mask.sum(dim=(1, 2, 3)) * v.shape[1]).clamp_min(eps)
    return num / den


def region_edge_energy(x, mask):
    gray = x.mean(dim=1, keepdim=True)
    dx = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return masked_mean(dx + dy, mask)


def region_frequency_energy(x, mask):
    low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    high = (x - low).pow(2)
    return masked_mean(high, mask)


def add_region_diffusion_noise(z0, mask, schedule, extra_keyword, extra_background):
    b = z0.shape[0]
    extra_keyword = extra_keyword.clamp(0, schedule.T - 1).long()
    extra_background = extra_background.clamp(0, schedule.T - 1).long()
    ab_k = schedule.alpha_bar[extra_keyword].view(b, 1, 1, 1)
    ab_b = schedule.alpha_bar[extra_background].view(b, 1, 1, 1)
    z_k = torch.sqrt(ab_k) * z0 + torch.sqrt(1.0 - ab_k) * torch.randn_like(z0)
    z_b = torch.sqrt(ab_b) * z0 + torch.sqrt(1.0 - ab_b) * torch.randn_like(z0)
    return (mask * z_k + (1.0 - mask) * z_b).clamp(0.0, 1.0)


def blend_region_denoise(z, denoised, mask, strength_keyword, strength_background):
    b = z.shape[0]
    s_k = strength_keyword.view(b, 1, 1, 1)
    s_b = strength_background.view(b, 1, 1, 1)
    y_k = (1.0 - s_k) * z + s_k * denoised
    y_b = (1.0 - s_b) * z + s_b * denoised
    return (mask * y_k + (1.0 - mask) * y_b).clamp(0.0, 1.0)
