"""Aggregate every finished run into tables, figures and reports/RESULTS.md.

    python scripts/make_report.py              # full results in runs/
    python scripts/make_report.py --quick      # smoke results in runs/quick/
    python scripts/make_report.py --no-tsne    # skip the slower embedding figures

Works on partial results, so it can be run while the grid is still going.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pcbssl.config import CLASS_NAMES, DATA_PROC, FIG_DIR, NUM_CLASSES, REPORTS_DIR, RUNS_DIR

ORDER = ["random", "imagenet", "simclr", "template_pos", "template_aware"]
LABEL = {"random": "Random init", "imagenet": "ImageNet", "simclr": "SimCLR (PCB)",
         "template_pos": "Template-positive (naive)", "template_aware": "Template-aware SimCLR (ours)"}
COLOR = {"random": "#8a8a8a", "imagenet": "#2a6fdb", "simclr": "#e08a00",
         "template_pos": "#b04fc4", "template_aware": "#1a9e5a"}
STYLE = {"template_pos": "--"}


def _init_key(name: str) -> str:
    return name.rsplit("_f", 1)[0]


def collect_cls(root: Path, mode: str) -> pd.DataFrame:
    rows = []
    for f in sorted((root / "cls" / mode).glob("*/metrics.json")):
        m = json.loads(f.read_text())
        su = m.get("test_seen_unseen_macro_f1", {})
        rows.append({"init": _init_key(f.parent.name), "fraction": m["fraction"], "seed": m["seed"],
                     "n_images": m["n_train_images"], "val_f1": m["val_macro_f1"],
                     "test_f1": m["test"]["macro_f1"], "test_acc": m["test"]["acc"],
                     "defect_f1": m["test"]["defect_macro_f1"],
                     "seen_f1": su.get("seen", np.nan), "unseen_f1": su.get("unseen", np.nan),
                     "per_class_f1": m["test"]["per_class_f1"], "confusion": m["test"]["confusion"],
                     "run": f.parent.name})
    return pd.DataFrame(rows)


def collect_det(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((root / "det").glob("*/metrics.json")):
        m = json.loads(f.read_text())
        t = m["test"]
        rows.append({"init": _init_key(f.parent.name), "fraction": m["fraction"], "seed": m["seed"],
                     "n_images": m["n_train_images"], "val_mAP50": m["val_mAP50"], "mAP50": t["mAP50"],
                     "mAP50_95": t["mAP50_95"], "F1@0.5": t["f1_at_best_50"]["f1"],
                     "deeppcb_mAP33": t["deeppcb_mAP33"], "deeppcb_F33": t["deeppcb_F33"]["f1"],
                     "seen_mAP50": t.get("seen_mAP50", np.nan), "unseen_mAP50": t.get("unseen_mAP50", np.nan),
                     "ap50_per_class": t["ap50_per_class"], "run": f.parent.name, "dir": str(f.parent)})
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, value: str) -> pd.DataFrame:
    g = df.groupby(["init", "fraction"])[value]
    s = pd.DataFrame({"mean": g.mean(), "min": g.min(), "max": g.max(), "n": g.count()}).reset_index()
    s["order"] = s["init"].map({k: i for i, k in enumerate(ORDER)})
    return s.sort_values(["order", "fraction"])


def curve_plot(df, value, ylabel, title, path, logx=True):
    s = summary(df, value)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for init in ORDER:
        d = s[s.init == init]
        if d.empty:
            continue
        x = d.fraction * 100
        ax.plot(x, d["mean"], marker="o", lw=2, ls=STYLE.get(init, "-"), color=COLOR[init], label=LABEL[init])
        if (d["n"] > 1).any():
            ax.fill_between(x, d["min"], d["max"], color=COLOR[init], alpha=0.15)
    if logx:
        ax.set_xscale("log")
    xs = sorted(s.fraction.unique() * 100)
    ax.set_xticks(xs, [f"{v:g}%" for v in xs])
    ax.minorticks_off()
    ax.set_xlabel("Labelled training images")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def table(df, value, digits=3) -> str:
    s = summary(df, value)
    fr = sorted(s.fraction.unique())
    lines = ["| Initialisation | " + " | ".join(f"{f * 100:g}% labels" for f in fr) + " |",
             "|---|" + "---|" * len(fr)]
    for init in ORDER:
        d = s[s.init == init].set_index("fraction")
        if d.empty:
            continue
        cells = []
        for f in fr:
            if f in d.index:
                r = d.loc[f]
                cell = f"{r['mean']:.{digits}f}"
                if r["n"] > 1:
                    cell += f" ({r['min']:.{digits}f} to {r['max']:.{digits}f})"
                cells.append(cell)
            else:
                cells.append("not run")
        lines.append(f"| {LABEL[init]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pretrain_plot(root: Path, path: Path) -> dict:
    hist = {}
    for mode in ("simclr", "template_pos", "template_aware"):
        f = root / "pretrain" / mode / "pretrain.json"
        if f.exists():
            hist[mode] = json.loads(f.read_text())["history"]
    if not hist:
        return {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for mode, h in hist.items():
        ax[0].plot([r["epoch"] for r in h], [r["loss"] for r in h], color=COLOR[mode], label=LABEL[mode])
        k = [r for r in h if "knn_macro_f1" in r]
        ax[1].plot([r["epoch"] for r in k], [r["knn_macro_f1"] for r in k], marker="o", color=COLOR[mode],
                   label=LABEL[mode])
    ax[0].set(xlabel="epoch", ylabel="contrastive loss", title="Pretraining loss")
    ax[1].set(xlabel="epoch", ylabel="val macro F1", title="kNN monitor on frozen features (k=20)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {m: h[-1] for m, h in hist.items()}


def confusion_plot(row, path, title):
    cm = np.asarray(row["confusion"], float)
    cmn = cm / cm.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_class_plot(df, col, names, path, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    inits = [i for i in ORDER if i in set(df.init)]
    w = 0.8 / max(1, len(inits))
    for k, init in enumerate(inits):
        v = np.mean(np.stack(df[df.init == init][col].apply(np.asarray).values), 0)
        ax.bar(np.arange(len(names)) + (k - (len(inits) - 1) / 2) * w, v, w, color=COLOR[init], label=LABEL[init])
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def seen_unseen_plot(df, seen_col, unseen_col, path, title, ylabel):
    inits = [i for i in ORDER if i in set(df.init)]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(inits))
    ax.bar(x - 0.2, [df[df.init == i][seen_col].mean() for i in inits], 0.4, label="seen board designs", color="#5a8fd8")
    ax.bar(x + 0.2, [df[df.init == i][unseen_col].mean() for i in inits], 0.4, label="unseen board designs", color="#e07b39")
    ax.set_xticks(x, [LABEL[i].replace(" (", "\n(") for i in inits], fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    lo = np.nanmin([df[seen_col].min(), df[unseen_col].min()])
    ax.set_ylim(max(0, lo - 0.1), 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tsne_plot(root: Path, path: Path, encoders: dict, n: int = 2500):
    from sklearn.manifold import TSNE

    from pcbssl.engine.features import embed
    from pcbssl.engine.utils import cls_cache, device
    from pcbssl.models.encoder import build_resnet18
    te = cls_cache("test")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(te["y"]))[:n]
    dev = device()
    items = [(k, v) for k, v in encoders.items()]
    fig, ax = plt.subplots(1, len(items), figsize=(4 * len(items), 4.2))
    ax = np.atleast_1d(ax)
    cmap = plt.get_cmap("tab10")
    for a, (name, init) in zip(ax, items):
        f = embed(build_resnet18(init).to(dev), te["x"][idx], dev).cpu().numpy()
        e = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(f)
        for c in range(NUM_CLASSES):
            m = te["y"][idx] == c
            a.scatter(e[m, 0], e[m, 1], s=3, color=cmap(c), label=CLASS_NAMES[c])
        a.set_title(LABEL[name], fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
    ax[-1].legend(markerscale=4, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle("t-SNE of frozen encoder features on test patches (no fine-tuning)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def gradcam_plot(root: Path, path: Path, inits: list[str], frac: float):
    from pcbssl.data.augment import normalize, to_input
    from pcbssl.engine.utils import cls_cache, device
    from pcbssl.eval.gradcam import GradCAM
    from pcbssl.models.encoder import PatchClassifier, build_resnet18
    te = cls_cache("test")
    dev = device()
    rng = np.random.default_rng(1)
    picks = [rng.choice(np.where(te["y"] == c)[0]) for c in range(1, NUM_CLASSES)]
    models = []
    for init in inits:
        f = root / "cls" / "finetune" / f"{init}_f{frac:g}_s0" / "model.pt"
        if f.exists():
            m = PatchClassifier(build_resnet18("random"), NUM_CLASSES)
            m.load_state_dict(torch.load(f, map_location="cpu", weights_only=False)["model"])
            models.append((init, m.to(dev)))
    if not models:
        return False
    x = normalize(to_input(torch.as_tensor(te["x"][picks]).to(dev)))
    fig, ax = plt.subplots(len(picks), 1 + len(models), figsize=(1.9 * (1 + len(models)), 1.9 * len(picks)))
    for r, i in enumerate(picks):
        ax[r, 0].imshow(te["x"][i], cmap="gray")
        ax[r, 0].set_ylabel(CLASS_NAMES[te["y"][i]], fontsize=9)
    ax[0, 0].set_title("patch", fontsize=9)
    for c, (init, m) in enumerate(models, start=1):
        cam = GradCAM(m, m.encoder.layer3)
        maps, pred = cam(x)
        cam.remove()
        for r, i in enumerate(picks):
            ax[r, c].imshow(te["x"][i], cmap="gray")
            ax[r, c].imshow(maps[r].cpu(), cmap="jet", alpha=0.45)
            ok = pred[r].item() == te["y"][i]
            ax[r, c].set_xlabel(f"pred: {CLASS_NAMES[pred[r]]}", fontsize=7, color="green" if ok else "red")
        ax[0, c].set_title(LABEL[init].replace(" (", "\n("), fontsize=8)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Grad-CAM on layer3, classifiers fine-tuned with {frac * 100:g}% labels", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def detection_examples(det: pd.DataFrame, path: Path, n: int = 3) -> str | None:
    from pcbssl.data.deeppcb import load_gray, load_splits
    from pcbssl.models.detector import build_detector
    from pcbssl.engine.utils import device
    best = det.sort_values("mAP50", ascending=False).iloc[0]
    f = Path(best["dir"]) / "model.pt"
    if not f.exists():
        return None
    dev = device()
    model = build_detector("random")
    model.load_state_dict(torch.load(f, map_location="cpu", weights_only=False)["model"])
    model.to(dev).eval()
    splits = load_splits()
    seen = {s.group for s in splits["train"]}
    test = splits["test"]
    picks = [next(s for s in test if s.group in seen), next(s for s in test if s.group not in seen), test[len(test) // 2]]
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 5.2))
    for a, s in zip(ax, picks[:n]):
        img = load_gray(s.test_path)
        x = torch.as_tensor(img).float().div(255).unsqueeze(0).expand(3, -1, -1).to(dev)
        with torch.no_grad():
            o = model([x])[0]
        a.imshow(img, cmap="gray")
        for (x1, y1, x2, y2) in s.boxes:
            a.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#2a6fdb", lw=2.5))
        for b, l, sc in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy()):
            if sc < 0.5:
                continue
            a.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#e0261b", lw=1.2))
            a.text(b[0], b[1] - 2, f"{CLASS_NAMES[l]} {sc:.2f}", color="#e0261b", fontsize=6)
        a.set_title(f"{s.id} ({'seen' if s.group in seen else 'unseen'} design)", fontsize=9)
        a.axis("off")
    fig.suptitle(f"Detections (red, score at least 0.5) vs ground truth (blue): {LABEL[best['init']]}, "
                 f"{best['fraction'] * 100:g}% labels", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return best["run"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-tsne", action="store_true")
    a = ap.parse_args()
    root = RUNS_DIR / "quick" if a.quick else RUNS_DIR
    fig_dir = FIG_DIR / "quick" if a.quick else FIG_DIR
    out_md = REPORTS_DIR / ("RESULTS_quick.md" if a.quick else "RESULTS.md")
    fig_dir.mkdir(parents=True, exist_ok=True)
    rel = lambda p: p.relative_to(REPORTS_DIR).as_posix()
    md = ["# Results", "", "Generated by `scripts/make_report.py` from the run folders. "
          "Values are on the official DeepPCB test split (500 images). Where several seeds exist, "
          "cells show the mean and, in brackets, the min to max range across seeds.", ""]

    stats = DATA_PROC / "stats.json"
    if stats.exists():
        s = json.loads(stats.read_text())
        md += ["## Data", "",
               f"* Images: {s['images']['train']} train, {s['images']['val']} validation, {s['images']['test']} test. "
               f"{s['unseen_test_images']} test images come from board designs never seen in training.",
               f"* Unlabelled pretraining patches: {s['ssl_patches']:,} "
               f"({s['ssl_grid_patches']:,} grid, {s['ssl_diff_guided_patches']:,} difference-guided).",
               f"* Label-free difference map recall of annotated defects: "
               f"{s['label_free_difference_recall_train']['loose_recall']:.1%} (loose), "
               f"{s['label_free_difference_recall_train']['strict_recall']:.1%} (strict).",
               "", "![pair](figures/dataset_pair.png)", ""]

    last = pretrain_plot(root, fig_dir / "pretrain_curves.png")
    if last:
        md += ["## Pretraining", "", f"![pretrain]({rel(fig_dir / 'pretrain_curves.png')})", "",
               "| Encoder | final loss | kNN val macro F1 |", "|---|---|---|"]
        for mode, r in last.items():
            md.append(f"| {LABEL[mode]} | {r['loss']:.3f} | {r.get('knn_macro_f1', float('nan')):.3f} |")
        md.append("")

    cls = collect_cls(root, "finetune")
    if not cls.empty:
        cls.drop(columns=["per_class_f1", "confusion"]).to_csv(fig_dir / "cls_finetune.csv", index=False)
        curve_plot(cls, "test_f1", "test macro F1 (7 classes)", "Stage A: patch classification",
                   fig_dir / "cls_label_efficiency.png")
        md += ["## Stage A: patch classification (fine-tuned)", "",
               f"![cls]({rel(fig_dir / 'cls_label_efficiency.png')})", "", "Test macro F1:", "",
               table(cls, "test_f1"), "", "Test macro F1 over the six defect classes only:", "",
               table(cls, "defect_f1"), ""]
        low = cls.fraction.min()
        lowdf = cls[cls.fraction == low]
        per_class_plot(lowdf, "per_class_f1", CLASS_NAMES, fig_dir / "cls_per_class.png",
                       f"Per-class test F1 at {low * 100:g}% labels", "F1")
        seen_unseen_plot(lowdf, "seen_f1", "unseen_f1", fig_dir / "cls_seen_unseen.png",
                         f"Generalisation to unseen board designs ({low * 100:g}% labels)", "macro F1")
        md += [f"![per-class]({rel(fig_dir / 'cls_per_class.png')})", "",
               f"![seen-unseen]({rel(fig_dir / 'cls_seen_unseen.png')})", ""]
        best_low = lowdf.groupby("init")["test_f1"].mean().idxmax()
        row = lowdf[(lowdf.init == best_low)].sort_values("seed").iloc[0]
        confusion_plot(row, fig_dir / "cls_confusion.png",
                       f"Confusion (row-normalised): {LABEL[best_low]}, {low * 100:g}% labels, seed {row['seed']}")
        md += [f"![confusion]({rel(fig_dir / 'cls_confusion.png')})", ""]
        if gradcam_plot(root, fig_dir / "gradcam.png", ["random", "imagenet", "simclr", "template_aware"],
                        0.1 if 0.1 in set(cls.fraction) else low):
            md += [f"![gradcam]({rel(fig_dir / 'gradcam.png')})", ""]

    lin = collect_cls(root, "linear")
    if not lin.empty:
        lin.drop(columns=["per_class_f1", "confusion"]).to_csv(fig_dir / "cls_linear.csv", index=False)
        curve_plot(lin, "test_f1", "test macro F1", "Linear probe on frozen encoders",
                   fig_dir / "cls_linear_probe.png")
        md += ["## Linear probe (frozen encoder)", "",
               "Measures representation quality directly: only a linear layer is trained.", "",
               f"![linear]({rel(fig_dir / 'cls_linear_probe.png')})", "", table(lin, "test_f1"), ""]

    det = collect_det(root)
    if not det.empty:
        det.drop(columns=["ap50_per_class"]).to_csv(fig_dir / "det_results.csv", index=False)
        curve_plot(det, "mAP50", "test mAP@0.5", "Stage B: defect detection (Faster R-CNN R18-FPN)",
                   fig_dir / "det_label_efficiency.png")
        md += ["## Stage B: detection (headline result)", "",
               f"![det]({rel(fig_dir / 'det_label_efficiency.png')})", "",
               "Test mAP@0.5:", "", table(det, "mAP50"), "",
               "Test mAP@0.5:0.95 (COCO-style):", "", table(det, "mAP50_95"), "",
               "DeepPCB benchmark protocol, F-score at IoU 0.33 at the best threshold:", "",
               table(det, "deeppcb_F33"), ""]
        low = det.fraction.min()
        dl = det[det.fraction == low]
        per_class_plot(dl, "ap50_per_class", CLASS_NAMES[1:], fig_dir / "det_per_class.png",
                       f"Per-class test AP@0.5 at {low * 100:g}% labels", "AP@0.5")
        seen_unseen_plot(det[det.fraction == low], "seen_mAP50", "unseen_mAP50", fig_dir / "det_seen_unseen.png",
                         f"Detection on seen vs unseen board designs ({low * 100:g}% labels)", "mAP@0.5")
        md += [f"![det-per-class]({rel(fig_dir / 'det_per_class.png')})", "",
               f"![det-seen]({rel(fig_dir / 'det_seen_unseen.png')})", ""]
        run = detection_examples(det, fig_dir / "det_examples.png")
        if run:
            md += [f"![examples]({rel(fig_dir / 'det_examples.png')})", ""]

    if not a.no_tsne:
        encs = {"random": "random", "imagenet": "imagenet"}
        for mode in ("simclr", "template_pos", "template_aware"):
            f = root / "pretrain" / mode / "encoder.pt"
            if f.exists():
                encs[mode] = str(f)
        tsne_plot(root, fig_dir / "tsne.png", encs)
        md += ["## Embeddings", "", f"![tsne]({rel(fig_dir / 'tsne.png')})", ""]

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
