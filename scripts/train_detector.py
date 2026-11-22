"""Stage B: train and evaluate Faster R-CNN (ResNet-18 FPN) on full DeepPCB images.

    python scripts/train_detector.py --init imagenet --fraction 1.0
    python scripts/train_detector.py --init runs/pretrain/template_aware/encoder.pt --fraction 0.1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbssl.config import RUNS_DIR
from pcbssl.engine.detect import train_detector
from pcbssl.models.encoder import init_name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="imagenet")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else RUNS_DIR / "det" / f"{init_name(a.init)}_f{a.fraction:g}_s{a.seed}"
    train_detector(out, a.init, a.fraction, a.seed, a.iters, a.batch_size, a.lr, eval_every=a.eval_every)


if __name__ == "__main__":
    main()
