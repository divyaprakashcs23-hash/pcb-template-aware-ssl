"""Contrastive pretraining on unlabelled PCB patches.

Modes: ``simclr``, ``template_pos`` (naive ablation) and ``template_aware`` (proposed).
See ``pcbssl.models.simclr`` for the definitions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data.augment import GPUAugment, normalize, to_input
from ..data.patches import CLEAN, DIFFERS
from ..models.encoder import build_resnet18
from ..models.simclr import SimCLR, nt_xent
from .features import knn_eval
from .utils import Timer, cls_cache, cosine_lr, device, save_json, set_lr, set_seed, ssl_cache

MODES = ("simclr", "template_pos", "template_aware")


def pretrain(out_dir: Path, mode: str = "simclr", epochs: int = 100, batch_size: int = 512,
             lr: float = 1e-3, weight_decay: float = 1e-4, tau: float = 0.2,
             p_template: float = 0.5, hard_weight: float = 1.0, warmup_epochs: int = 5,
             seed: int = 0, knn_every: int = 10, max_patches: int | None = None,
             init: str = "random", log=print) -> dict:
    assert mode in MODES, mode
    set_seed(seed)
    dev = device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = ssl_cache()
    n = len(data["test"]) if max_patches is None else min(max_patches, len(data["test"]))
    sel = np.random.default_rng(seed).permutation(len(data["test"]))[:n]
    P_test = torch.as_tensor(data["test"][sel]).to(dev)
    P_temp = torch.as_tensor(data["temp"][sel]).to(dev)
    P_state = torch.as_tensor(data["state"][sel]).to(dev)
    log(f"[pretrain:{mode}] {n} patches | clean {(P_state == CLEAN).float().mean():.1%} "
        f"| differs {(P_state == DIFFERS).float().mean():.1%}")

    # kNN monitor (labels used for monitoring only, never for training)
    tr, va = cls_cache("train"), cls_cache("val")
    rng = np.random.default_rng(0)
    bank = rng.permutation(len(tr["y"]))[:20000]

    model = SimCLR(build_resnet18(init)).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    aug = GPUAugment.ssl()
    steps_per_epoch = n // batch_size
    total = epochs * steps_per_epoch
    warmup = warmup_epochs * steps_per_epoch

    start_epoch, history = 0, []
    last = out_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, history = ck["epoch"], ck["history"]
        log(f"[pretrain:{mode}] resumed from epoch {start_epoch}")

    timer = Timer()
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    for epoch in range(start_epoch, epochs):
        model.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        tot_loss, tot_acc, tot_gap, n_gap = 0.0, 0.0, 0.0, 0
        for it in range(steps_per_epoch):
            step = epoch * steps_per_epoch + it
            set_lr(opt, cosine_lr(step, total, lr, warmup))
            idx = perm[it * batch_size:(it + 1) * batch_size]
            xt, xp, st = to_input(P_test[idx]), to_input(P_temp[idx]), P_state[idx]
            B = len(idx)
            coin = torch.rand(B, device=dev) < p_template
            hard = torch.zeros(B, dtype=torch.bool, device=dev)
            if mode == "simclr":
                use_t = torch.zeros_like(coin)
            elif mode == "template_pos":
                use_t = coin
            else:
                use_t = coin & (st == CLEAN)
                hard = st == DIFFERS
            src2 = torch.where(use_t[:, None, None, None], xp, xt)
            views = [aug(xt), aug(src2)]
            if hard.any():
                views.append(aug(xp[hard]))
            x = normalize(torch.cat(views)).contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                z = model(x)
            z1, z2 = z[:B], z[B:2 * B]
            zh = None
            if hard.any():
                zh = torch.zeros_like(z1)
                zh[hard] = z[2 * B:]
            loss, stats = nt_xent(z1, z2, tau, zh, hard if hard.any() else None, hard_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item()
            tot_acc += stats["pos_acc"]
            if "hard_gap" in stats:
                tot_gap += stats["hard_gap"]
                n_gap += 1
        rec = {"epoch": epoch + 1, "loss": tot_loss / steps_per_epoch,
               "pos_acc": tot_acc / steps_per_epoch, "time_s": round(timer(), 1)}
        if n_gap:
            rec["hard_gap"] = tot_gap / n_gap
        if knn_every and ((epoch + 1) % knn_every == 0 or epoch + 1 == epochs):
            rec.update(knn_eval(model.encoder, tr["x"][bank], tr["y"][bank], va["x"], va["y"], dev))
        history.append(rec)
        log(f"[pretrain:{mode}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                             for k, v in rec.items()))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "history": history}, last)

    meta = {"mode": mode, "epochs": epochs, "batch_size": batch_size, "lr": lr, "tau": tau,
            "p_template": p_template, "hard_weight": hard_weight, "n_patches": n, "seed": seed,
            "init": init}
    torch.save({"encoder": model.encoder.state_dict(), "meta": meta}, out_dir / "encoder.pt")
    save_json({"meta": meta, "history": history}, out_dir / "pretrain.json")
    last.unlink(missing_ok=True)
    return {"meta": meta, "history": history}
