import itertools
import torch


class KeywordActionSpace:
    def __init__(self, cfg):
        self.actions = list(itertools.product(
            list(cfg["extra_t_keyword"]),
            list(cfg["extra_t_background"]),
            list(cfg["denoise_keyword"]),
            list(cfg["denoise_background"]),
            list(cfg["risk_threshold"]),
        ))

    def __len__(self):
        return len(self.actions)

    def decode(self, action_idx, device=None):
        vals = [self.actions[int(i)] for i in action_idx.detach().cpu().tolist()]
        dev = device or action_idx.device

        e_k = torch.tensor([v[0] for v in vals], dtype=torch.long, device=dev)
        e_b = torch.tensor([v[1] for v in vals], dtype=torch.long, device=dev)
        d_k = torch.tensor([v[2] for v in vals], dtype=torch.float32, device=dev)
        d_b = torch.tensor([v[3] for v in vals], dtype=torch.float32, device=dev)
        th = torch.tensor([v[4] for v in vals], dtype=torch.float32, device=dev)

        return e_k, e_b, d_k, d_b, th

    def mask(self, batch_size, device, max_extra_t, keyword_score=None, guard_cfg=None):
        """
        Build valid-action mask.

        Basic constraints:
          1. extra_t_keyword >= extra_t_background
          2. denoise_keyword <= denoise_background
          3. extra_t values do not exceed max_extra_t

        Dynamic keyword guard:
          If public keyword score on z0 is high, block weak protection actions.
          This does not use true labels.
        """
        mask = torch.ones(batch_size, len(self.actions), dtype=torch.bool, device=device)

        if guard_cfg is None:
            guard_cfg = {}

        guard_enabled = bool(guard_cfg.get("enabled", False))
        score_trigger = float(guard_cfg.get("score_trigger", 0.35))
        min_extra = int(guard_cfg.get("min_extra_keyword_when_detected", 350))
        max_denoise = float(guard_cfg.get("max_denoise_keyword_when_detected", 0.10))
        max_threshold = float(guard_cfg.get("max_threshold_when_detected", 0.05))

        high_keyword = None
        if guard_enabled and keyword_score is not None:
            high_keyword = keyword_score.detach().to(device).view(batch_size) >= score_trigger

        for j, (e_k, e_b, d_k, d_b, th) in enumerate(self.actions):
            allowed = torch.ones(batch_size, dtype=torch.bool, device=device)

            # Basic hard constraints.
            if e_k > max_extra_t or e_b > max_extra_t:
                allowed &= False

            # Dynamic keyword guard.
            # If keyword appears likely in z0, disallow weak keyword protection.
            # Means: if keyword present: protect strongly
            #        else: do not over - protect
            if high_keyword is not None:
                # For detected keyword samples, require keyword protection.
                if e_k < e_b:
                    allowed &= ~high_keyword

                if d_k > d_b:
                    allowed &= ~high_keyword

                strong_enough = (
                        e_k >= min_extra
                        and d_k <= max_denoise
                        and th <= max_threshold
                )

                if not strong_enough:
                    allowed &= ~high_keyword

                # For non-detected samples, prevent unnecessary keyword blurring.
                low_keyword = ~high_keyword

                max_extra_not = int(guard_cfg.get("max_extra_keyword_when_not_detected", 0))
                min_denoise_not = float(guard_cfg.get("min_denoise_keyword_when_not_detected", 0.50))

                if e_k > max_extra_not:
                    allowed &= ~low_keyword

                if d_k < min_denoise_not:
                    allowed &= ~low_keyword


            else:
                # Fallback for generic always-sensitive keyword such as "face".
                # Keyword region should receive at least as much extra noise as background.
                if e_k < e_b:
                    allowed &= False

                # Keyword region should receive no stronger denoising than background.
                if d_k > d_b:
                    allowed &= False

            mask[:, j] &= allowed

        # Fallback: if a sample has no valid action, allow the strongest protective action.
        empty = ~mask.any(dim=1)
        if empty.any():
            best_idx = 0
            best_score = None

            for j, (e_k, e_b, d_k, d_b, th) in enumerate(self.actions):
                # Prefer more keyword noise, weaker keyword denoise, stricter threshold.
                score = (e_k, -d_k, -th, -e_b, d_b)
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = j

            mask[empty, best_idx] = True

        return mask