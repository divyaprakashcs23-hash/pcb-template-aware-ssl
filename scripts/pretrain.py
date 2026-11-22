"""Contrastive pretraining on unlabelled DeepPCB patches.

    python scripts/pretrain.py --mode simclr --epochs 100
    python scripts/pretrain.py --mode template_aware --epochs 100
    python scripts/pretrain.py --mode template_pos --epochs 100   # naive ablation
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbssl.config import RUNS_DIR
from pcbssl.engine.pretrain import MODES, pretrain


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, default="simclr")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--p-template", type=float, default=0.5)
    ap.add_argument("--hard-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--knn-every", type=int, default=10)
    ap.add_argument("--max-patches", type=int, default=None)
    ap.add_argument("--init", default="random", help="random or imagenet starting weights")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else RUNS_DIR / "pretrain" / a.mode
    pretrain(out, a.mode, a.epochs, a.batch_size, a.lr, tau=a.tau, p_template=a.p_template,
             hard_weight=a.hard_weight, seed=a.seed, knn_every=a.knn_every,
             max_patches=a.max_patches, init=a.init)


if __name__ == "__main__":
    main()
