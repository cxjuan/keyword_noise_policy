import torch
import torch.nn.functional as F


def mse(x, y):
    return F.mse_loss(x, y, reduction='none').flatten(1).mean(dim=1)


def psnr(x, y, eps=1e-8):
    m = mse(x, y).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / m)


def edge_energy(x):
    gray = x.mean(dim=1, keepdim=True)
    dx = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return (dx + dy).mean(dim=(1, 2, 3))


def blur_proxy(x):
    return 1.0 / (edge_energy(x) + 1e-4)


def frequency_energy(x):
    low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    high = x - low
    return high.pow(2).mean(dim=(1, 2, 3))
