import random
import numpy as np
import torch
import yaml
from pathlib import Path


def load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
