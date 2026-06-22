from pathlib import Path
import torch
import torch.nn.functional as F
from .classifier import SmallDigitCNN
from .region_ops import masked_mean
import open_clip
import torch.nn.functional as F

def _soft_center_ellipse(x, rx=0.34, ry=0.42, cy=0.48, softness=0.08):
    b, _, h, w = x.shape
    device = x.device
    yy = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1)
    xx = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w)
    dist = ((xx - 0.5) / rx) ** 2 + ((yy - cy) / ry) ** 2
    return torch.sigmoid((1.0 - dist) / softness).expand(b, 1, h, w).clamp(0, 1)


def _hair_prior_mask(x, y_max=0.48, x_margin=0.16, softness=0.08):
    b, _, h, w = x.shape
    device = x.device
    yy = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1)
    xx = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w)
    top = torch.sigmoid((float(y_max) - yy) / float(softness))
    x_left = torch.sigmoid((xx - float(x_margin)) / float(softness))
    x_right = torch.sigmoid((1.0 - float(x_margin) - xx) / float(softness))
    return (top * x_left * x_right).expand(b, 1, h, w).clamp(0, 1)


def _normalize_mask(m, top_ratio=0.30, smooth=True, dilate=True):
    b = m.shape[0]
    flat = m.flatten(1)
    k = max(1, int(flat.shape[1] * (1.0 - float(top_ratio))))
    thresh = flat.kthvalue(k, dim=1).values.view(b, 1, 1, 1)
    mask = (m >= thresh).float()
    if smooth:
        mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)
        mask = (mask > 0.25).float()
    if dilate:
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return mask.clamp(0, 1)


def foreground_mask_from_noised(z, top_ratio=0.45, smooth=True, dilate=True):
    gray = z.mean(dim=1, keepdim=True)
    b = gray.shape[0]
    flat = gray.flatten(1)

    k = max(1, int(flat.shape[1] * (1.0 - top_ratio)))
    thresh = flat.kthvalue(k, dim=1).values.view(b, 1, 1, 1)

    mask = (gray >= thresh).float()

    if smooth:
        mask = torch.nn.functional.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)
        mask = (mask > 0.25).float()

    if dilate:
        mask = torch.nn.functional.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

    return mask.clamp(0, 1)

class MNISTDigitKeyword:
    def __init__(self, cfg, device):
        kcfg = cfg['keyword']
        # self.target_digit = int(kcfg['target_digit'])
        if "target_digits" in kcfg:
            self.target_digits = [int(v) for v in kcfg["target_digits"]]
        elif "target_digit" in kcfg:
            self.target_digits = [int(kcfg["target_digit"])]
        else:
            raise ValueError("keyword config needs target_digit or target_digits")

        self.top_ratio = float(kcfg.get('saliency_top_ratio', 0.30))
        self.smooth = bool(kcfg.get('saliency_smooth', True))
        self.dilate = bool(kcfg.get('saliency_dilate', True))
        self.foreground_top_ratio = float(kcfg.get("foreground_top_ratio", 0.45))
        self.device = device
        self.classifier = SmallDigitCNN(in_channels=1, num_classes=10).to(device)
        ckpt = Path(kcfg['classifier_ckpt'])
        if not ckpt.exists():
            raise FileNotFoundError(f'Digit classifier checkpoint not found: {ckpt}. Run train_keyword_classifier.py first.')
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        self.classifier.load_state_dict(state)
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)

    def encode(self, z0):
        z = z0.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.classifier(z)
            probs = logits.softmax(dim=1)

            target_idx = torch.tensor(self.target_digits, device=z.device, dtype=torch.long)

            # Keyword score = probability of any target digit, e.g. P(5 or 6).
            score = probs[:, target_idx].sum(dim=1).clamp(0.0, 1.0)

            # Attribution for all target digits together.
            selected = logits[:, target_idx].sum()
            grad = torch.autograd.grad(selected, z, retain_graph=False, create_graph=False)[0]

            sal = grad.abs().mean(dim=1, keepdim=True)
        sal_mask = _normalize_mask(sal, self.top_ratio, self.smooth, self.dilate)
        fg_mask = foreground_mask_from_noised(
            z0,
            top_ratio=self.foreground_top_ratio,
            smooth=True,
            dilate=True,
        )

        mask = torch.clamp(sal_mask.detach() + fg_mask.detach(), 0, 1)

        return score.detach(), mask.detach(), {'probs': probs.detach()}

    def score_only(self, y):
        with torch.no_grad():
            probs = self.classifier(y).softmax(dim=1)
            target_idx = torch.tensor(
                self.target_digits,
                device=y.device,
                dtype=torch.long
            )
            return probs[:, target_idx].sum(dim=1).clamp(0.0, 1.0)


class FacePriorKeyword:
    def __init__(self, cfg, device):
        self.cfg = cfg['keyword']

    def encode(self, z0):
        mask = _soft_center_ellipse(z0, float(self.cfg.get('ellipse_rx', 0.34)), float(self.cfg.get('ellipse_ry', 0.42)), float(self.cfg.get('ellipse_cy', 0.48)), float(self.cfg.get('ellipse_softness', 0.08)))
        score = torch.ones(z0.shape[0], device=z0.device)
        return score, mask, {}

    def score_only(self, y):
        return torch.ones(y.shape[0], device=y.device)


class FaceBlackHairPriorKeyword:
    def __init__(self, cfg, device):
        self.cfg = cfg['keyword']

    def encode(self, z0):
        face = _soft_center_ellipse(z0, float(self.cfg.get('ellipse_rx', 0.34)), float(self.cfg.get('ellipse_ry', 0.42)), float(self.cfg.get('ellipse_cy', 0.48)), float(self.cfg.get('ellipse_softness', 0.08)))
        hair = _hair_prior_mask(z0, float(self.cfg.get('hair_y_max', 0.48)), float(self.cfg.get('hair_x_margin', 0.16)), float(self.cfg.get('hair_softness', 0.08)))
        mode = str(self.cfg.get('mask_mode', 'face_plus_hair')).lower()
        mask = hair if mode == 'hair_only' else torch.clamp(face + hair, 0, 1)
        gray = z0.mean(dim=1, keepdim=True)
        darkness = 1.0 - masked_mean(gray, hair).clamp(0, 1)
        score = darkness.clamp(0, 1)
        return score, mask, {'hair_mask': hair, 'face_mask': face}

    def score_only(self, y):
        hair = _hair_prior_mask(y, float(self.cfg.get('hair_y_max', 0.48)), float(self.cfg.get('hair_x_margin', 0.16)), float(self.cfg.get('hair_softness', 0.08)))
        gray = y.mean(dim=1, keepdim=True)
        return (1.0 - masked_mean(gray, hair).clamp(0, 1)).clamp(0, 1)



class CLIPTextKeyword:
    def __init__(self, cfg, device):
        self.cfg = cfg["keyword"]
        self.device = device

        model_name = self.cfg.get("clip_model", "ViT-B-32")
        pretrained = self.cfg.get("clip_pretrained", "laion2b_s34b_b79k")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.prompts = self.cfg.get("prompts", None)
        if self.prompts is None:
            prompt = self.cfg.get("prompt", None)
            if prompt is None:
                raise ValueError("clip_text keyword needs prompts or prompt")
            self.prompts = [prompt]

        self.negative_prompts = self.cfg.get("negative_prompts", ["an unrelated image"])

        self.saliency_top_ratio = float(self.cfg.get("saliency_top_ratio", 0.35))
        self.saliency_smooth = bool(self.cfg.get("saliency_smooth", True))
        self.saliency_dilate = bool(self.cfg.get("saliency_dilate", True))
        self.score_scale = float(self.cfg.get("score_scale", 20.0))
        self.score_margin = float(self.cfg.get("score_margin", 0.0))

        with torch.no_grad():
            pos_tokens = self.tokenizer(self.prompts).to(device)
            neg_tokens = self.tokenizer(self.negative_prompts).to(device)

            self.pos_text_features = self.model.encode_text(pos_tokens)
            self.neg_text_features = self.model.encode_text(neg_tokens)

            self.pos_text_features = F.normalize(self.pos_text_features, dim=-1)
            self.neg_text_features = F.normalize(self.neg_text_features, dim=-1)

    def _clip_input(self, x):
        # CLIP expects 224x224 RGB normalized images.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=x.device,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=x.device,
        ).view(1, 3, 1, 1)

        return (x - mean) / std

    def _score_from_image(self, x):
        x_clip = self._clip_input(x)

        img_feat = self.model.encode_image(x_clip)
        img_feat = F.normalize(img_feat, dim=-1)

        pos_sim = img_feat @ self.pos_text_features.T
        neg_sim = img_feat @ self.neg_text_features.T

        pos_score = pos_sim.mean(dim=1)
        neutral_score = neg_sim.mean(dim=1)

        score_scale = float(self.cfg.get("score_scale", 25.0))
        score_margin = float(self.cfg.get("score_margin", 0.02))

        # Relative CLIP score: target prompt vs negative prompt.
        # Generic text-keyword score.
        # The margin simply prevents weak prompt matches
        # from activating protection for all faces.
        score = torch.sigmoid(
            score_scale * (pos_score - neutral_score - score_margin)
        )

        return score.clamp(0.0, 1.0)


    def encode(self, z0):
        # z0 is already minimum-DP noised.
        with torch.enable_grad():
            z = z0.detach().clone().requires_grad_(True)
            score = self._score_from_image(z)

            selected = score.sum()
            grad = torch.autograd.grad(
                selected,
                z,
                retain_graph=False,
                create_graph=False,
            )[0]

            sal = (grad.abs() * z.abs()).mean(dim=1, keepdim=True)
            mask = _normalize_mask(
                sal,
                top_ratio=self.saliency_top_ratio,
                smooth=self.saliency_smooth,
                dilate=self.saliency_dilate,
            )

        return score.detach(), mask.detach(), {}

    def score_only(self, y):
        with torch.no_grad():
            return self._score_from_image(y).detach()


def build_keyword_module(cfg, device):
    typ = str(cfg["keyword"]["type"]).lower()

    if typ == "mnist_digit":
        return MNISTDigitKeyword(cfg, device)

    if typ == "face_prior":
        return FacePriorKeyword(cfg, device)

    if typ == "face_blackhair_prior":
        return FaceBlackHairPriorKeyword(cfg, device)

    if typ == "clip_text":
        return CLIPTextKeyword(cfg, device)

    raise ValueError(f"Unsupported keyword type: {typ}")
