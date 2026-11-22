"""ResNet-18 encoders with the four initialisations compared in this project.

The architecture is the unmodified torchvision ResNet-18 so that one set of weights loads
both into the patch classifier and into the Faster R-CNN FPN backbone. Inputs are
grayscale patches replicated to 3 channels.

``init`` can be ``random``, ``imagenet``, or a path to an encoder checkpoint saved by
``pcbssl.engine.pretrain`` (SimCLR or template-aware SimCLR).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

FEAT_DIM = 512


def build_resnet18(init: str = "random") -> nn.Module:
    if init == "imagenet":
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        m = resnet18(weights=None)
    m.fc = nn.Identity()
    if init not in ("random", "imagenet"):
        state = load_encoder_state(init)
        missing, unexpected = m.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("fc.")]
        if missing or unexpected:
            raise RuntimeError(f"Encoder checkpoint mismatch. missing={missing} unexpected={unexpected}")
    return m


def load_encoder_state(path: str | Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("encoder", ckpt)
    return {k: v for k, v in state.items() if not k.startswith("fc.")}


def init_name(init: str) -> str:
    """Short human-readable name for an init string (used in run names and plots)."""
    if init in ("random", "imagenet"):
        return init
    p = Path(init)
    return p.parent.name if p.name.startswith("encoder") else p.stem


class PatchClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(FEAT_DIM, num_classes))

    def forward(self, x):
        return self.head(self.encoder(x))
