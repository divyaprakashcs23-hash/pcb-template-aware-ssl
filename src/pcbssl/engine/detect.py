"""Stage B: Faster R-CNN (ResNet-18 FPN) defect detection at a given label fraction.

All images are cached on the GPU as uint8 (about 400 MB for the whole dataset), and the
dihedral augmentations are applied to images and boxes there, so no DataLoader is needed.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from ..config import NUM_CLASSES
from ..data.deeppcb import Sample, load_gray, load_splits, seen_groups
from ..data.sampler import label_fraction_subset
from ..eval.metrics import detection_report
from ..models.detector import build_detector
from .utils import Timer, cosine_lr, device, save_json, set_lr, set_seed


class ImageCache:
    def __init__(self, samples: list[Sample], dev, with_template: bool = False):
        self.samples = samples
        self.images = torch.as_tensor(np.stack([load_gray(s.test_path) for s in samples])).to(dev)
        self.templates = None
        if with_template:
            self.templates = torch.as_tensor(np.stack([load_gray(s.temp_path) for s in samples])).to(dev)
        self.boxes = [torch.as_tensor(s.boxes_np, device=dev) for s in samples]
        self.labels = [torch.as_tensor(s.labels_np, device=dev) for s in samples]

    def __len__(self):
        return len(self.samples)


def dihedral(img: torch.Tensor, boxes: torch.Tensor, k: int, flip: bool):
    """Rotate a square (H,W) image by k*90 degrees counter-clockwise, then optionally flip
    horizontally, and transform xyxy boxes to match."""
    S = img.shape[-1]
    b = boxes.clone()
    for _ in range(k % 4):
        img = torch.rot90(img, 1, dims=(-2, -1))
        x1, y1, x2, y2 = b.unbind(1)
        b = torch.stack([y1, S - x2, y2, S - x1], 1)
    if flip:
        img = torch.flip(img, dims=(-1,))
        x1, y1, x2, y2 = b.unbind(1)
        b = torch.stack([S - x2, y1, S - x1, y2], 1)
    return img, b


def _to_model_input(img_u8: torch.Tensor) -> torch.Tensor:
    return img_u8.float().div(255.0).unsqueeze(0).expand(3, -1, -1)


@torch.no_grad()
def predict_images(model, cache: ImageCache, bs: int = 8) -> list[dict]:
    model.eval()
    dev = cache.images.device
    preds = []
    for i in range(0, len(cache), bs):
        imgs = [_to_model_input(cache.images[j]) for j in range(i, min(i + bs, len(cache)))]
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(imgs)
        for o in out:
            preds.append({k: o[k].float().cpu().numpy() for k in ("boxes", "scores", "labels")})
    return preds


def ground_truth(cache: ImageCache) -> list[dict]:
    return [{"boxes": s.boxes_np, "labels": s.labels_np} for s in cache.samples]


def evaluate(model, cache: ImageCache, seen: set[str] | None = None) -> dict:
    preds, gts = predict_images(model, cache), ground_truth(cache)
    rep = detection_report(preds, gts, NUM_CLASSES)
    if seen is not None:
        m = np.array([s.group in seen for s in cache.samples])
        for name, mask in (("seen", m), ("unseen", ~m)):
            if mask.any():
                idx = np.where(mask)[0]
                r = detection_report([preds[i] for i in idx], [gts[i] for i in idx], NUM_CLASSES)
                rep[f"{name}_mAP50"] = r["mAP50"]
                rep[f"{name}_n_images"] = int(mask.sum())
    return rep


def train_detector(out_dir: Path, init: str = "random", fraction: float = 1.0, seed: int = 0,
                   iters: int = 4000, batch_size: int = 8, lr: float = 0.01, weight_decay: float = 1e-4,
                   eval_every: int = 500, warmup: int = 300, save_model: bool = True, log=print) -> dict:
    set_seed(seed)
    dev = device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    train_s = label_fraction_subset(splits["train"], fraction, seed)
    tr, va, te = ImageCache(train_s, dev), ImageCache(splits["val"], dev), ImageCache(splits["test"], dev)
    log(f"[det] init={init} frac={fraction} seed={seed} | {len(tr)} train images, "
        f"{sum(len(s.boxes) for s in train_s)} boxes")

    model = build_detector(init).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    g = torch.Generator()
    g.manual_seed(seed)
    timer = Timer()
    best, best_state, curve, run_loss = -1.0, None, [], 0.0
    perm, ptr = torch.randperm(len(tr), generator=g), 0
    for it in range(iters):
        model.train()
        set_lr(opt, cosine_lr(it, iters, lr, warmup))
        if ptr + batch_size > len(perm):
            perm, ptr = torch.randperm(len(tr), generator=g), 0
        idx = perm[ptr:ptr + batch_size].tolist() if len(tr) >= batch_size else \
            torch.randint(0, len(tr), (batch_size,), generator=g).tolist()
        ptr += batch_size
        imgs, targets = [], []
        for j in idx:
            k = int(torch.randint(0, 4, (1,), generator=g))
            flip = bool(torch.rand(1, generator=g) < 0.5)
            im, bx = dihedral(tr.images[j], tr.boxes[j], k, flip)
            imgs.append(_to_model_input(im))
            targets.append({"boxes": bx, "labels": tr.labels[j]})
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            losses = model(imgs, targets)
            loss = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        scaler.step(opt)
        scaler.update()
        run_loss += loss.item()
        if (it + 1) % eval_every == 0 or it + 1 == iters:
            vr = evaluate(model, va)
            rec = {"iter": it + 1, "loss": run_loss / eval_every, "val_mAP50": vr["mAP50"],
                   "time_s": round(timer(), 1)}
            run_loss = 0.0
            curve.append(rec)
            log(f"[det] {rec}")
            if vr["mAP50"] > best:
                best, best_state = vr["mAP50"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test = evaluate(model, te, seen_groups(splits))
    result = {"init": init, "fraction": fraction, "seed": seed, "iters": iters,
              "n_train_images": len(tr), "val_mAP50": best, "test": test, "curve": curve,
              "time_s": round(timer(), 1)}
    if save_model:
        torch.save({"model": model.state_dict(), "init": init}, out_dir / "model.pt")
    save_json(result, out_dir / "metrics.json")
    log(f"[det] done val_mAP50={best:.4f} test_mAP50={test['mAP50']:.4f} "
        f"deeppcb_F33={test['deeppcb_F33']['f1']:.4f} ({result['time_s']}s)")
    return result
