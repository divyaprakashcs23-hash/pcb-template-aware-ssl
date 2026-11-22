"""Grad-CAM (Selvaraju et al., 2017) for the patch classifier.

The class score is back-propagated to a ResNet stage (layer3 by default in this project, since layer4 is only 2x2 on a 64x64 patch). Channel weights are the
spatially averaged gradients, and the map is the ReLU of the weighted activation sum,
upsampled to the input size.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        self.h1 = layer.register_forward_hook(self._fwd)
        self.h2 = layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o

    def _bwd(self, m, gi, go):
        self.grads = go[0]

    def __call__(self, x: torch.Tensor, cls: torch.Tensor | None = None):
        """x: normalised (B,3,H,W). Returns (cams (B,H,W) in [0,1], predicted classes)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        pred = logits.argmax(1)
        target = pred if cls is None else cls
        self.model.zero_grad(set_to_none=True)
        logits.gather(1, target[:, None]).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach(), pred.detach()

    def remove(self):
        self.h1.remove()
        self.h2.remove()
