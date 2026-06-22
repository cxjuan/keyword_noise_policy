import torch
import torch.nn.functional as F


class PublicDenoiser:
    def __call__(self, z: torch.Tensor, strength: torch.Tensor):
        raise NotImplementedError


class SimplePublicDenoiser(PublicDenoiser):
    def __call__(self, z: torch.Tensor, strength: torch.Tensor):
        smooth = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)
        s = strength.view(-1, 1, 1, 1)
        out = (1.0 - s) * z + s * smooth
        return out.clamp(0.0, 1.0)


class PublicDiffusionDenoiser(PublicDenoiser):
    def __init__(self, model=None):
        self.model = model

    def __call__(self, z: torch.Tensor, strength: torch.Tensor):
        if self.model is None:
            return SimplePublicDenoiser()(z, strength)
        raise NotImplementedError('Plug a public diffusion denoiser here.')
