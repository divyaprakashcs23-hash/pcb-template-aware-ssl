"""Batched GPU augmentations for grayscale PCB patches.

Each sample in a batch gets its own random parameters. Running on the GPU keeps a batch-512
SimCLR step fast on a laptop GPU without relying on Windows DataLoader workers.

PCB-specific choices: colour jitter is meaningless on binarised images, so photometric
augmentation is limited to contrast, brightness, blur and sensor noise. Defects are
orientation invariant, so all eight dihedral transforms are allowed.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import IMAGENET_MEAN, IMAGENET_STD


def to_input(x_u8: torch.Tensor) -> torch.Tensor:
    """uint8 (B,H,W) or (B,1,H,W) -> float (B,1,H,W) in [0,1]."""
    if x_u8.dim() == 3:
        x_u8 = x_u8.unsqueeze(1)
    return x_u8.float().div_(255.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """float (B,1,H,W) in [0,1] -> ImageNet-normalised (B,3,H,W)."""
    x = x.expand(-1, 3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class GPUAugment(nn.Module):
    def __init__(self, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3), max_translate=0.0,
                 dihedral=True, contrast=0.4, brightness=0.2, blur_p=0.5,
                 blur_sigma=(0.1, 1.5), noise_p=0.5, noise_std=0.06):
        super().__init__()
        self.scale, self.ratio, self.max_translate = scale, ratio, max_translate
        self.dihedral = dihedral
        self.contrast, self.brightness = contrast, brightness
        self.blur_p, self.blur_sigma = blur_p, blur_sigma
        self.noise_p, self.noise_std = noise_p, noise_std

    @classmethod
    def ssl(cls) -> "GPUAugment":
        """Strong views for contrastive pretraining."""
        return cls()

    @classmethod
    def finetune(cls) -> "GPUAugment":
        """Light augmentation for supervised fine-tuning."""
        return cls(scale=(0.8, 1.0), ratio=(0.9, 1.1), max_translate=0.06, contrast=0.2,
                   brightness=0.1, blur_p=0.2, blur_sigma=(0.1, 0.8), noise_p=0.3, noise_std=0.03)

    def _geometry(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        dev = x.device
        area = torch.empty(B, device=dev).uniform_(*self.scale)
        logr = torch.empty(B, device=dev).uniform_(math.log(self.ratio[0]), math.log(self.ratio[1]))
        r = torch.exp(logr)
        sw = torch.sqrt(area * r).clamp(max=1.0)
        sh = torch.sqrt(area / r).clamp(max=1.0)
        cx = (torch.rand(B, device=dev) * 2 - 1) * (1 - sw)
        cy = (torch.rand(B, device=dev) * 2 - 1) * (1 - sh)
        if self.max_translate > 0:
            cx = cx + (torch.rand(B, device=dev) * 2 - 1) * self.max_translate
            cy = cy + (torch.rand(B, device=dev) * 2 - 1) * self.max_translate
        A = torch.zeros(B, 2, 2, device=dev)
        A[:, 0, 0], A[:, 1, 1] = sw, sh
        if self.dihedral:
            k = torch.randint(0, 4, (B,), device=dev)
            ang = k.float() * (math.pi / 2)
            c, s = torch.cos(ang).round(), torch.sin(ang).round()
            R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
            flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
            Fm = torch.diag_embed(torch.stack([flip, torch.ones_like(flip)], -1))
            A = A @ R @ Fm
        theta = torch.cat([A, torch.stack([cx, cy], -1).unsqueeze(-1)], -1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        k = 7
        sig = torch.empty(B, device=x.device).uniform_(*self.blur_sigma)
        t = torch.arange(k, device=x.device).float() - k // 2
        g = torch.exp(-(t[None] ** 2) / (2 * sig[:, None] ** 2))
        g = g / g.sum(1, keepdim=True)
        apply = torch.rand(B, device=x.device) < self.blur_p
        ident = torch.zeros_like(g)
        ident[:, k // 2] = 1
        g = torch.where(apply[:, None], g, ident)
        xr = x.transpose(0, 1)  # (1,B,H,W), one group per sample
        xr = F.conv2d(F.pad(xr, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(B, 1, 1, k), groups=B)
        xr = F.conv2d(F.pad(xr, (0, 0, k // 2, k // 2), mode="reflect"), g.view(B, 1, k, 1), groups=B)
        return xr.transpose(0, 1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: float (B,1,H,W) in [0,1]. Returns augmented float (B,1,H,W) in [0,1]."""
        x = self._geometry(x)
        B = x.shape[0]
        dev = x.device
        c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * self.contrast
        b = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * self.brightness
        x = (x - 0.5) * c + 0.5 + b
        if self.blur_p > 0:
            x = self._blur(x)
        if self.noise_p > 0:
            m = (torch.rand(B, 1, 1, 1, device=dev) < self.noise_p).float()
            x = x + m * torch.randn_like(x) * self.noise_std
        return x.clamp_(0, 1)
