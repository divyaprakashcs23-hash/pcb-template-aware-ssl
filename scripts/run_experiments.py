"""Run the full experiment grid. Resumable: finished runs (metrics.json present) are skipped.

    python scripts/run_experiments.py                 # everything, in the recommended order
    python scripts/run_experiments.py --stage cls     # one stage only
    python scripts/run_experiments.py --quick         # tiny smoke version of the whole grid

Stages, in order:
    pretrain   SimCLR and template-aware SimCLR encoders
    cls        Stage A fine-tuning grid (core initialisations)
    det        Stage B detection grid
    ablation   naive template-positive encoder plus its classification runs
    linear     frozen-encoder linear probes for every encoder
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbssl.config import RUNS_DIR

ENCODERS = {
    "simclr": RUNS_DIR / "pretrain" / "simclr" / "encoder.pt",
    "template_aware": RUNS_DIR / "pretrain" / "template_aware" / "encoder.pt",
    "template_pos": RUNS_DIR / "pretrain" / "template_pos" / "encoder.pt",
}
CORE_INITS = ["random", "imagenet", "simclr", "template_aware"]
CLS_FRACTIONS = [0.02, 0.10, 0.25, 1.00]
DET_FRACTIONS = [0.10, 0.25, 1.00]
SEEDS = [0, 1, 2]

FULL = dict(pre_epochs=60, cls_steps=1500, det_iters=2500, det_eval=500, seeds=SEEDS, det_seeds=[0])
QUICK = dict(pre_epochs=1, cls_steps=60, det_iters=40, det_eval=20, seeds=[0], det_seeds=[0])


def resolve(init: str) -> str:
    return str(ENCODERS[init]) if init in ENCODERS else init


def log(msg: str, path: Path):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("all", "pretrain", "cls", "det", "ablation", "linear"), default="all")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = QUICK if args.quick else FULL
    root = RUNS_DIR / ("quick" if args.quick else "")
    root.mkdir(parents=True, exist_ok=True)
    logfile = root / "experiments.log"
    L = lambda m: log(m, logfile)
    if args.quick:
        for k in ENCODERS:
            ENCODERS[k] = root / "pretrain" / k / "encoder.pt"

    from pcbssl.engine.classify import train_classifier
    from pcbssl.engine.detect import train_detector
    from pcbssl.engine.pretrain import pretrain

    def do_pretrain(mode):
        if ENCODERS[mode].exists():
            L(f"skip pretrain {mode}")
            return
        L(f"start pretrain {mode}")
        pretrain(ENCODERS[mode].parent, mode, epochs=cfg["pre_epochs"], log=L,
                 max_patches=4096 if args.quick else None)

    def do_cls(init, frac, seed, mode="finetune"):
        out = root / "cls" / mode / f"{init}_f{frac:g}_s{seed}"
        if (out / "metrics.json").exists():
            return
        L(f"start cls {mode} {init} f={frac} s={seed}")
        train_classifier(out, resolve(init), frac, seed, mode=mode, steps=cfg["cls_steps"],
                         eval_every=max(20, cfg["cls_steps"] // 6), save_model=(seed == 0), log=L)

    def do_det(init, frac, seed):
        out = root / "det" / f"{init}_f{frac:g}_s{seed}"
        if (out / "metrics.json").exists():
            return
        L(f"start det {init} f={frac} s={seed}")
        train_detector(out, resolve(init), frac, seed, iters=cfg["det_iters"],
                       eval_every=cfg["det_eval"], warmup=min(300, cfg["det_iters"] // 5), log=L)

    st = args.stage
    if st in ("all", "pretrain"):
        do_pretrain("simclr")
        do_pretrain("template_aware")
    if st in ("all", "cls"):
        for frac in CLS_FRACTIONS:
            for seed in cfg["seeds"]:
                for init in CORE_INITS:
                    do_cls(init, frac, seed)
    if st in ("all", "det"):
        for frac in DET_FRACTIONS:
            for init in CORE_INITS:
                for seed in cfg["det_seeds"]:
                    do_det(init, frac, seed)
    if st in ("all", "ablation"):
        do_pretrain("template_pos")
        for frac in CLS_FRACTIONS:
            for seed in cfg["seeds"]:
                do_cls("template_pos", frac, seed)
    if st in ("all", "linear"):
        inits = CORE_INITS + ["template_pos"]
        for frac in CLS_FRACTIONS:
            for seed in cfg["seeds"]:
                for init in inits:
                    if init in ENCODERS and not ENCODERS[init].exists():
                        continue
                    do_cls(init, frac, seed, mode="linear")
    L(f"stage {st} finished")


if __name__ == "__main__":
    main()
