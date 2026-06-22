import torch
from .diffusion_noise import DiffusionSchedule
from .denoiser import SimplePublicDenoiser
from .actions_keyword import KeywordActionSpace
from .keyword_modules import build_keyword_module
from .region_ops import add_region_diffusion_noise, blend_region_denoise, region_edge_energy, region_frequency_energy
from .metrics import psnr


class KeywordDiPPOImageEnv:
    def __init__(self, cfg, device='cpu'):
        self.cfg = cfg
        dcfg = cfg['diffusion']
        self.schedule = DiffusionSchedule(T=dcfg['T'], beta_start=dcfg['beta_start'], beta_end=dcfg['beta_end'], device=device)
        self.min_t = int(dcfg['min_t'])
        self.max_extra_t = int(dcfg.get('max_extra_t', 500))
        self.delta = float(dcfg['delta'])
        self.sensitivity_l2 = float(dcfg['sensitivity_l2'])
        self.target_epsilon = float(dcfg['target_epsilon'])
        self.action_space = KeywordActionSpace(cfg['keyword_actions'])
        self.denoiser = SimplePublicDenoiser()
        self.keyword_module = build_keyword_module(cfg, device)

    def make_obs_from_floor(self, x):
        b = x.shape[0]
        t0 = torch.full((b,), self.min_t, dtype=torch.long, device=x.device)
        z0 = self.schedule.diffuse(x, t0).clamp(0.0, 1.0)
        eps_floor = self.schedule.epsilon_upper_bound(t0, self.sensitivity_l2, self.delta)
        keyword_score, mask, aux = self.keyword_module.encode(z0)

        kcfg = self.cfg.get("keyword", {})
        detect_thr = float(kcfg.get("detect_score_threshold", 0.55))
        zero_mask = bool(kcfg.get("zero_mask_when_not_detected", True))

        keyword_detected = keyword_score >= detect_thr

        # if a face does not match the prompt (gole), its keyword mask becomes zero.
        # Then the face should not be blurred by keyword-region protection.
        if zero_mask:
            gate = keyword_detected.float().view(-1, 1, 1, 1)
            mask = mask * gate

        bg = 1.0 - mask
        mask_area = mask.mean(dim=(1, 2, 3))
        kw_edge = region_edge_energy(z0, mask)
        bg_edge = region_edge_energy(z0, bg)
        kw_freq = region_frequency_energy(z0, mask)
        bg_freq = region_frequency_energy(z0, bg)
        edge_ratio = (kw_edge / (bg_edge + 1e-6)).clamp(0, 10) / 10.0
        freq_ratio = (kw_freq / (bg_freq + 1e-6)).clamp(0, 10) / 10.0
        obs = torch.stack([
            keyword_score.clamp(0, 1), mask_area.clamp(0, 1),
            kw_edge, bg_edge, kw_freq, bg_freq,
            edge_ratio, freq_ratio,
            eps_floor / max(self.target_epsilon, 1e-6),
            torch.ones((b,), device=x.device),
        ], dim=1)
        return obs, z0, mask, keyword_score

    def action_mask(self, obs):
        keyword_score = obs[:, 0]
        return self.action_space.mask(
            obs.shape[0],
            obs.device,
            self.max_extra_t,
            keyword_score=keyword_score,
            guard_cfg=self.cfg.get("keyword_guard", {}),
        )

    def step(self, x, action_idx):
        b = x.shape[0]
        obs, z0, mask, keyword_score_before = self.make_obs_from_floor(x)
        e_k, e_b, d_k, d_b, threshold = self.action_space.decode(action_idx, device=x.device)
        z = add_region_diffusion_noise(z0=z0, mask=mask, schedule=self.schedule, extra_keyword=e_k, extra_background=e_b)
        base_denoised = self.denoiser(z, torch.ones(b, device=x.device))
        candidate = blend_region_denoise(z=z, denoised=base_denoised, mask=mask, strength_keyword=d_k, strength_background=d_b)
        bg = 1.0 - mask
        global_utility = psnr(candidate, x) / 40.0
        bg_dist = ((candidate - x).pow(2) * bg).sum(dim=(1, 2, 3)) / (bg.sum(dim=(1, 2, 3)) * x.shape[1]).clamp_min(1e-6)
        bg_utility = 1.0 / (1.0 + 20.0 * bg_dist)
        keyword_change = ((candidate - z0).pow(2) * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) * x.shape[1]).clamp_min(1e-6)
        kw_freq_z0 = region_frequency_energy(z0, mask)
        kw_freq_cand = region_frequency_energy(candidate, mask)
        kw_hf_gain = (kw_freq_cand - kw_freq_z0).clamp_min(0.0)
        keyword_score_after = self.keyword_module.score_only(candidate).detach()

        keyword_score_candidate = self.keyword_module.score_only(candidate).detach()

        risk = (
                keyword_change
                + 0.5 * kw_hf_gain
                + 0.5 * keyword_score_candidate
        )

        kcfg = self.cfg.get("keyword", {})
        detect_thr = float(kcfg.get("detect_score_threshold", 0.55))
        release_limit = float(kcfg.get("release_score_limit", 0.45))


        keyword_detected = keyword_score_before >= detect_thr
        keyword_presence = keyword_detected.float()
        keyword_still_visible = keyword_score_candidate >= release_limit
        keyword_score_candidate = self.keyword_module.score_only(candidate).detach()

        rejected = keyword_detected & ((risk > threshold) | keyword_still_visible)

        released = candidate.clone()
        released[rejected] = z[rejected]

        keyword_score_released = self.keyword_module.score_only(released).detach()
        keyword_leak = keyword_presence * keyword_score_released


        extra_k_norm = e_k.float() / max(1.0, float(self.max_extra_t))
        extra_b_norm = e_b.float() / max(1.0, float(self.max_extra_t))
        rcfg = self.cfg['reward']
        reward = (
                rcfg["global_utility_weight"] * global_utility
                + rcfg["background_utility_weight"] * bg_utility
                - rcfg["keyword_confidence_weight"] * keyword_leak
                - rcfg["risk_weight"] * keyword_presence * risk
                - rcfg["keyword_denoise_penalty"] * keyword_presence * d_k
                - rcfg["background_noise_penalty"] * extra_b_norm
                + rcfg["keyword_noise_bonus"] * keyword_presence * extra_k_norm
                - rcfg["reject_penalty_weight"] * rejected.float()
        )
        info = {
            'z0': z0.detach(), 'mask': mask.detach(), 'noised': z.detach(), 'released': released.detach(), 'candidate': candidate.detach(),
            'risk': risk.detach(), 'threshold': threshold.detach(), 'rejected': rejected.detach(),
            'extra_keyword': e_k.detach(), 'extra_background': e_b.detach(),
            'denoise_keyword': d_k.detach(), 'denoise_background': d_b.detach(),
            'keyword_score_before': keyword_score_before.detach(), 'keyword_score_after': keyword_score_after.detach(),
            'global_utility': global_utility.detach(), 'bg_utility': bg_utility.detach(),
            "keyword_score_candidate": keyword_score_candidate.detach(),
            "keyword_score_released": keyword_score_released.detach(),
            "keyword_leak": keyword_leak.detach(),
        }
        return obs, reward, info
