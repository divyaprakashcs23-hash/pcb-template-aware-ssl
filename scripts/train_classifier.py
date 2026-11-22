"""Stage A: train and evaluate the 7-class patch classifier.

    python scripts/train_classifier.py --init imagenet --fraction 0.1 --seed 0
    python scripts/train_classifier.py --init runs/pretrain/template_aware/encoder.pt --fraction 0.1
    python scripts/train_classifier.py --init random --mode linear
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbssl.config import RUNS_DIR
from pcbssl.engine.classify import train_classifier
from pcbssl.models.encoder import init_name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="random")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("finetune", "linear"), default="finetune")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else RUNS_DIR / "cls" / a.mode / f"{init_name(a.init)}_f{a.fraction:g}_s{a.seed}"
    train_classifier(out, a.init, a.fraction, a.seed, a.mode, a.steps, a.batch_size, a.lr,
                     class_weighted=not a.no_class_weights)


if __name__ == "__main__":
    main()
