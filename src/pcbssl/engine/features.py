"""Feature extraction and the kNN monitor used during pretraining."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..config import NUM_CLASSES
from ..data.augment import normalize, to_input
from ..eval.metrics import classification_report


@torch.no_grad()
def embed(encoder, x_u8: np.ndarray | torch.Tensor, dev, bs: int = 1024) -> torch.Tensor:
    was = encoder.training
    encoder.eval()
    out = []
    for i in range(0, len(x_u8), bs):
        xb = torch.as_tensor(x_u8[i:i + bs]).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(encoder(normalize(to_input(xb))).float())
    encoder.train(was)
    return torch.cat(out)


@torch.no_grad()
def knn_eval(encoder, bank_x, bank_y, query_x, query_y, dev, k: int = 20, t: float = 0.07) -> dict:
    """Weighted kNN classification on frozen features (Wu et al., 2018). Monitoring only."""
    fb = F.normalize(embed(encoder, bank_x, dev), dim=1)
    fq = F.normalize(embed(encoder, query_x, dev), dim=1)
    yb = torch.as_tensor(bank_y, device=dev)
    preds = []
    for i in range(0, len(fq), 2048):
        sim = fq[i:i + 2048] @ fb.t()
        w, idx = sim.topk(k, dim=1)
        w = (w / t).exp()
        votes = torch.zeros(len(w), NUM_CLASSES, device=dev)
        votes.scatter_add_(1, yb[idx], w)
        preds.append(votes.argmax(1))
    pred = torch.cat(preds).cpu().numpy()
    r = classification_report(query_y, pred, NUM_CLASSES)
    return {"knn_acc": r["acc"], "knn_macro_f1": r["macro_f1"]}
