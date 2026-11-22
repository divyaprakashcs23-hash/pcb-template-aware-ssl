"""Small shared helpers: seeding, schedules, JSON I/O, data caches."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..config import DATA_PROC
from ..data.patches import load_npz


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_lr(step: int, total: int, base: float, warmup: int, final_frac: float = 0.0) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr * g.get("lr_mult", 1.0)


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def cls_cache(split: str) -> dict:
    return load_npz(DATA_PROC / f"cls_{split}.npz")


def ssl_cache() -> dict:
    return load_npz(DATA_PROC / "ssl_train.npz")


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return time.time() - self.t0
