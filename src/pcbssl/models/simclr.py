"""SimCLR and the template-aware contrastive objective.

Standard SimCLR (Chen et al., 2020): two augmented views of a patch are positives, all other
views in the batch are negatives, NT-Xent loss.

Template-aware variants use the aligned defect-free template that every DeepPCB pair (and
every real AOI line) provides. The patch-level difference state is computed without labels.

* ``template_pos``  (naive): with probability p the second view comes from the template crop,
  regardless of content. For a defective patch this asks the encoder to map "with defect"
  and "without defect" to the same point, i.e. to become blind to defects. Kept as an ablation.
* ``template_aware`` (proposed):
    - clean patches (test == template): with probability p the second view is the template
      crop. This is a real positive and adds capture-level nuisance variation.
    - differing patches: both views come from the test crop, and the template crop at the
      same location is added as an extra *hard negative* for that anchor. The only thing that
      separates them is the defect, so the encoder is pushed to represent it.
    - ambiguous patches (edge noise only): plain SimCLR.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import FEAT_DIM


class Projector(nn.Module):
    def __init__(self, in_dim: int = FEAT_DIM, hidden: int = 512, out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.projector = Projector(out=proj_dim)

    def forward(self, x):
        return self.projector(self.encoder(x))


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2,
            hard_neg: torch.Tensor | None = None, hard_mask: torch.Tensor | None = None,
            hard_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """NT-Xent over 2N views with an optional per-anchor hard negative.

    z1, z2: (N, d) projections of the two views. hard_neg: (N, d) projections of the template
    crop for each sample, used only where hard_mask is True, for both anchors of that sample.
    hard_weight > 1 up-weights the hard negative in the denominator (added as log weight).
    """
    N = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], 0).float(), dim=1)
    sim = z @ z.t() / tau
    eye = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))
    target = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    logits = sim
    if hard_neg is not None and hard_mask is not None and hard_mask.any():
        h = F.normalize(hard_neg.float(), dim=1)
        h2 = torch.cat([h, h], 0)
        hn = (z * h2).sum(1, keepdim=True) / tau
        if hard_weight != 1.0:
            hn = hn + torch.log(torch.tensor(hard_weight, device=z.device))
        m2 = torch.cat([hard_mask, hard_mask], 0).unsqueeze(1)
        hn = hn.masked_fill(~m2, float("-inf"))
        logits = torch.cat([sim, hn], 1)
    loss = F.cross_entropy(logits, target)
    with torch.no_grad():
        acc = (logits.argmax(1) == target).float().mean().item()
        stats = {"pos_acc": acc}
        if logits.shape[1] > 2 * N:
            pos = sim.gather(1, target[:, None]).squeeze(1)
            m = torch.cat([hard_mask, hard_mask], 0)
            stats["hard_gap"] = (pos[m] - logits[m, -1]).mean().item() * tau if m.any() else 0.0
    return loss, stats
