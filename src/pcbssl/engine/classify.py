"""Stage A: 7-class patch classification at a given label fraction.

Protocol (identical for every initialisation):
* labelled images = nested, group-stratified subset of the training images
* fixed number of optimisation steps, independent of the fraction
* model selection on validation macro F1, test evaluated once with the selected weights
* class-weighted cross-entropy (inverse square-root frequency) for the background imbalance
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NUM_CLASSES
from ..data.augment import GPUAugment, normalize, to_input
from ..data.deeppcb import load_splits, seen_groups
from ..data.sampler import label_fraction_subset
from ..eval.metrics import classification_report
from ..models.encoder import PatchClassifier, build_resnet18
from .features import embed
from .utils import Timer, cls_cache, cosine_lr, device, save_json, set_lr, set_seed


def labelled_mask(cache: dict, fraction: float, seed: int) -> np.ndarray:
    splits = load_splits()
    keep = {s.id for s in label_fraction_subset(splits["train"], fraction, seed)}
    ids = cache["ids"][cache["img_idx"]]
    return np.isin(ids, list(keep))


@torch.no_grad()
def predict(model, x_u8, dev, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x_u8), bs):
        xb = torch.as_tensor(x_u8[i:i + bs]).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(model(normalize(to_input(xb))).float().softmax(1).cpu())
    return torch.cat(out).numpy()


def _seen_unseen(test: dict, pred: np.ndarray) -> dict:
    seen = seen_groups(load_splits())
    g = test["groups"][test["img_idx"]]
    m = np.isin(g, list(seen))
    out = {}
    for name, mask in (("seen", m), ("unseen", ~m)):
        if mask.any():
            out[name] = classification_report(test["y"][mask], pred[mask], NUM_CLASSES)["macro_f1"]
    return out


def train_classifier(out_dir: Path, init: str = "random", fraction: float = 1.0, seed: int = 0,
                     mode: str = "finetune", steps: int = 2000, batch_size: int = 256,
                     lr: float = 5e-4, weight_decay: float = 0.05, eval_every: int = 200,
                     class_weighted: bool = True, save_model: bool = True, log=print) -> dict:
    set_seed(seed)
    dev = device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va, te = cls_cache("train"), cls_cache("val"), cls_cache("test")
    m = labelled_mask(tr, fraction, seed)
    X, Y = tr["x"][m], tr["y"][m]
    counts = np.bincount(Y, minlength=NUM_CLASSES).astype(np.float64)
    log(f"[cls] init={init} frac={fraction} seed={seed} mode={mode} | {m.sum()} patches "
        f"from {len(np.unique(tr['img_idx'][m]))} images | counts={counts.astype(int).tolist()}")
    weight = None
    if class_weighted:
        w = (counts.sum() / np.maximum(counts, 1) / NUM_CLASSES) ** 0.5
        weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)

    encoder = build_resnet18(init)
    timer = Timer()
    if mode == "linear":
        result = _linear_probe(encoder.to(dev), X, Y, va, te, weight, dev, seed, log)
    else:
        model = PatchClassifier(encoder, NUM_CLASSES).to(dev).to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
        aug = GPUAugment.finetune()
        Xg, Yg = torch.as_tensor(X).to(dev), torch.as_tensor(Y).to(dev)
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        best, best_state, curve = -1.0, None, []
        perm, ptr = torch.randperm(len(Xg), device=dev, generator=g), 0
        for step in range(steps):
            model.train()
            set_lr(opt, cosine_lr(step, steps, lr, warmup=min(100, steps // 10)))
            if ptr + batch_size > len(perm):
                perm, ptr = torch.randperm(len(Xg), device=dev, generator=g), 0
            idx = perm[ptr:ptr + batch_size]
            ptr += batch_size
            x = normalize(aug(to_input(Xg[idx]))).contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = F.cross_entropy(model(x).float(), Yg[idx], weight=weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if (step + 1) % eval_every == 0 or step + 1 == steps:
                vr = classification_report(va["y"], predict(model, va["x"], dev).argmax(1), NUM_CLASSES)
                curve.append({"step": step + 1, "loss": loss.item(), "val_macro_f1": vr["macro_f1"]})
                if vr["macro_f1"] > best:
                    best, best_state = vr["macro_f1"], copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        prob = predict(model, te["x"], dev)
        pred = prob.argmax(1)
        result = {"val_macro_f1": best, "test": classification_report(te["y"], pred, NUM_CLASSES),
                  "test_seen_unseen_macro_f1": _seen_unseen(te, pred), "curve": curve}
        if save_model:
            torch.save({"model": model.state_dict(), "init": init}, out_dir / "model.pt")
    result.update({"init": init, "fraction": fraction, "seed": seed, "mode": mode, "steps": steps,
                   "n_train_patches": int(m.sum()),
                   "n_train_images": int(len(np.unique(tr["img_idx"][m]))), "time_s": round(timer(), 1)})
    save_json(result, out_dir / "metrics.json")
    log(f"[cls] done val_f1={result['val_macro_f1']:.4f} test_f1={result['test']['macro_f1']:.4f} "
        f"acc={result['test']['acc']:.4f} ({result['time_s']}s)")
    return result


def _linear_probe(encoder, X, Y, va, te, weight, dev, seed, log, epochs: int = 100) -> dict:
    """Frozen-encoder linear evaluation: the standard measure of representation quality."""
    ftr = embed(encoder, X, dev)
    fva, fte = embed(encoder, va["x"], dev), embed(encoder, te["x"], dev)
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    ftr, fva, fte = (ftr - mu) / sd, (fva - mu) / sd, (fte - mu) / sd
    yt = torch.as_tensor(Y, device=dev)
    head = nn.Linear(ftr.shape[1], NUM_CLASSES).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = -1.0, None
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(ftr), device=dev, generator=g)
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]
            loss = F.cross_entropy(head(ftr[idx]), yt[idx], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                f1 = classification_report(va["y"], head(fva).argmax(1).cpu().numpy(), NUM_CLASSES)["macro_f1"]
            if f1 > best:
                best, best_state = f1, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    with torch.no_grad():
        pred = head(fte).argmax(1).cpu().numpy()
    return {"val_macro_f1": best, "test": classification_report(te["y"], pred, NUM_CLASSES),
            "test_seen_unseen_macro_f1": _seen_unseen(te, pred)}
