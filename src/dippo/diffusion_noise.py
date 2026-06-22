import math
import torch


class DiffusionSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device='cpu'):
        self.T = int(T)
        betas = torch.linspace(beta_start, beta_end, self.T, device=device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def diffuse(self, x: torch.Tensor, t: torch.Tensor):
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        eps = torch.randn_like(x)
        return torch.sqrt(ab) * x + torch.sqrt(1.0 - ab) * eps

    def epsilon_upper_bound(self, t: torch.Tensor, sensitivity_l2: float, delta: float):
        # Approximate Gaussian mechanism bound for M(x)=sqrt(ab)*x + sqrt(1-ab)*N(0,I).
        ab = self.alpha_bar[t].clamp(1e-12, 1.0 - 1e-12)
        ratio = torch.sqrt(ab) * sensitivity_l2 / torch.sqrt(1.0 - ab)
        return ratio * math.sqrt(2.0 * math.log(1.25 / float(delta)))
