"""Build splits and patch caches from the raw DeepPCB clone.

    python scripts/prepare_data.py            # full build
    python scripts/prepare_data.py --download # clone DeepPCB first if missing

Outputs in data/processed/: splits.json, ssl_train.npz, cls_{train,val,test}.npz, stats.json,
and dataset figures in reports/figures/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pcbssl.config import CLASS_NAMES, DATA_PROC, DATA_RAW, DEEPPCB_ROOT, FIG_DIR
from pcbssl.data.deeppcb import load_gray, make_splits, save_splits, seen_groups
from pcbssl.data.patches import (AMBIGUOUS, CLEAN, DIFFERS, build_cls_patches, build_ssl_patches,
                                 difference_maps, save_npz)
from pcbssl.engine.utils import save_json

DEEPPCB_URL = "https://github.com/tangsanli5201/DeepPCB.git"


def download():
    if DEEPPCB_ROOT.exists():
        print(f"DeepPCB already present at {DEEPPCB_ROOT}")
        return
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", DEEPPCB_URL, str(DATA_RAW / "DeepPCB")])


def label_free_recall(samples) -> dict:
    """How many annotated defects the label-free difference map touches."""
    hit_l, hit_s, n = 0, 0, 0
    for s in samples:
        loose, strict = difference_maps(load_gray(s.test_path), load_gray(s.temp_path))
        for x1, y1, x2, y2 in s.boxes:
            hit_l += loose[y1:y2, x1:x2].any()
            hit_s += strict[y1:y2, x1:x2].any()
            n += 1
    return {"loose_recall": hit_l / n, "strict_recall": hit_s / n, "n_boxes": n}


def figures(splits, cls_tr, ssl):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    s = splits["train"][0]
    test, temp = load_gray(s.test_path), load_gray(s.temp_path)
    loose, strict = difference_maps(test, temp)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    ax[0].imshow(test, cmap="gray")
    for (x1, y1, x2, y2), c in zip(s.boxes, s.labels):
        ax[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.5))
        ax[0].text(x1, y1 - 3, CLASS_NAMES[c], color="red", fontsize=8)
    ax[0].set_title(f"Tested image {s.id} with annotations")
    ax[1].imshow(temp, cmap="gray")
    ax[1].set_title("Defect-free template")
    ax[2].imshow(loose, cmap="Greys")
    ax[2].imshow(np.ma.masked_where(~strict, strict), cmap="autumn", alpha=0.9)
    ax[2].set_title("Label-free difference map (grey loose, red strict)")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dataset_pair.png", dpi=130)
    plt.close(fig)

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(len(CLASS_NAMES), 8, figsize=(9, 8.5))
    for c in range(len(CLASS_NAMES)):
        idx = rng.choice(np.where(cls_tr["y"] == c)[0], 8, replace=False)
        for j, i in enumerate(idx):
            ax[c, j].imshow(cls_tr["x"][i], cmap="gray", vmin=0, vmax=255)
            ax[c, j].axis("off")
        ax[c, 0].set_title(CLASS_NAMES[c], fontsize=9, loc="left")
    fig.suptitle("Labelled 64x64 patches per class")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_patches.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(3, 8, figsize=(9, 3.8))
    for r, st in enumerate((CLEAN, DIFFERS, AMBIGUOUS)):
        idx = rng.choice(np.where(ssl["state"] == st)[0], 4, replace=False)
        for j, i in enumerate(idx):
            ax[r, 2 * j].imshow(ssl["test"][i], cmap="gray", vmin=0, vmax=255)
            ax[r, 2 * j + 1].imshow(ssl["temp"][i], cmap="gray", vmin=0, vmax=255)
        for a in ax[r]:
            a.axis("off")
        ax[r, 0].set_title(["clean", "differs", "ambiguous"][r] + " (test | template)", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ssl_patch_states.png", dpi=130)
    plt.close(fig)

    counts = {split: Counter(c for s_ in splits[split] for c in s_.labels) for split in splits}
    fig, ax = plt.subplots(figsize=(7, 3.5))
    w = 0.27
    for k, split in enumerate(("train", "val", "test")):
        ax.bar(np.arange(1, 7) + (k - 1) * w, [counts[split][c] for c in range(1, 7)], w, label=split)
    ax.set_xticks(range(1, 7), CLASS_NAMES[1:])
    ax.set_ylabel("defect boxes")
    ax.legend()
    ax.set_title("DeepPCB defect counts per split")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_counts.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--bg-ratio", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.download:
        download()
    if not DEEPPCB_ROOT.exists():
        sys.exit(f"DeepPCB not found at {DEEPPCB_ROOT}. Run with --download.")

    splits = make_splits(args.val_frac, args.seed)
    save_splits(splits)
    seen = seen_groups(splits)
    print({k: len(v) for k, v in splits.items()})

    ssl = build_ssl_patches(splits["train"], seed=args.seed)
    save_npz(DATA_PROC / "ssl_train.npz", ssl)
    caches = {}
    for i, split in enumerate(("train", "val", "test")):
        caches[split] = build_cls_patches(splits[split], seed=args.seed + i, bg_ratio=args.bg_ratio)
        save_npz(DATA_PROC / f"cls_{split}.npz", caches[split])

    st = np.bincount(ssl["state"], minlength=3)
    stats = {
        "images": {k: len(v) for k, v in splits.items()},
        "boxes": {k: int(sum(len(s.boxes) for s in v)) for k, v in splits.items()},
        "seen_groups": sorted(seen),
        "unseen_test_groups": sorted({s.group for s in splits["test"]} - seen),
        "unseen_test_images": int(sum(s.group not in seen for s in splits["test"])),
        "ssl_patches": int(len(ssl["state"])),
        "ssl_grid_patches": int((ssl["kind"] == 0).sum()),
        "ssl_diff_guided_patches": int((ssl["kind"] == 1).sum()),
        "ssl_state_fraction": {"clean": st[0] / st.sum(), "differs": st[1] / st.sum(),
                               "ambiguous": st[2] / st.sum()},
        "cls_patches": {k: np.bincount(v["y"], minlength=len(CLASS_NAMES)).tolist() for k, v in caches.items()},
        "label_free_difference_recall_train": label_free_recall(splits["train"]),
    }
    save_json(stats, DATA_PROC / "stats.json")
    figures(splits, caches["train"], ssl)
    import json
    print(json.dumps(stats, indent=1, default=float))


if __name__ == "__main__":
    main()
