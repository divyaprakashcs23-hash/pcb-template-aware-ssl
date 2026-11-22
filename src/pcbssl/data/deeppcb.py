"""DeepPCB parsing, splits and image loading.

Each DeepPCB sample is an aligned pair: a tested image with defects and a defect-free
template of the same board location, both 640x640 binarised grayscale. Annotations are
``x1 y1 x2 y2 type`` per line with type in 1..6.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import DATA_PROC, DEEPPCB_ROOT


@dataclass
class Sample:
    id: str
    group: str
    test_path: str
    temp_path: str
    boxes: list = field(default_factory=list)   # [[x1, y1, x2, y2], ...]
    labels: list = field(default_factory=list)  # [1..6, ...]

    @property
    def boxes_np(self) -> np.ndarray:
        return np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)

    @property
    def labels_np(self) -> np.ndarray:
        return np.asarray(self.labels, dtype=np.int64)


def parse_annotation(path: Path) -> tuple[list, list]:
    boxes, labels = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) != 5:
            continue
        x1, y1, x2, y2, c = map(int, parts)
        if x2 <= x1 or y2 <= y1 or not 1 <= c <= 6:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(c)
    return boxes, labels


def load_official_split(split: str, root: Path = DEEPPCB_ROOT) -> list[Sample]:
    """Load the official ``trainval`` or ``test`` list shipped with DeepPCB."""
    root = Path(root)
    samples = []
    for line in (root / f"{split}.txt").read_text().splitlines():
        if not line.strip():
            continue
        img_rel, ann_rel = line.split()
        img = root / img_rel
        sid = img.stem
        boxes, labels = parse_annotation(root / ann_rel)
        samples.append(Sample(
            id=sid,
            group=img_rel.split("/")[0],
            test_path=str(img.with_name(f"{sid}_test.jpg")),
            temp_path=str(img.with_name(f"{sid}_temp.jpg")),
            boxes=boxes,
            labels=labels,
        ))
    return samples


def make_splits(val_frac: float = 0.1, seed: int = 0, root: Path = DEEPPCB_ROOT) -> dict[str, list[Sample]]:
    """Official test split, plus a validation split carved from trainval, stratified by group.

    Validation is used only for model selection. Test is evaluated once per run, at the end.
    """
    trainval = load_official_split("trainval", root)
    test = load_official_split("test", root)
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[Sample]] = {}
    for s in trainval:
        by_group.setdefault(s.group, []).append(s)
    train, val = [], []
    for g in sorted(by_group):
        items = sorted(by_group[g], key=lambda s: s.id)
        perm = rng.permutation(len(items))
        n_val = max(1, int(round(val_frac * len(items))))
        val += [items[i] for i in perm[:n_val]]
        train += [items[i] for i in perm[n_val:]]
    return {"train": sorted(train, key=lambda s: s.id),
            "val": sorted(val, key=lambda s: s.id),
            "test": test}


def seen_groups(splits: dict[str, list[Sample]]) -> set[str]:
    """Board groups present in training. Test images from other groups are unseen designs."""
    return {s.group for s in splits["train"]}


def save_splits(splits: dict[str, list[Sample]], path: Path = DATA_PROC / "splits.json") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: [asdict(s) for s in v] for k, v in splits.items()}, indent=1))


def load_splits(path: Path = DATA_PROC / "splits.json") -> dict[str, list[Sample]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/prepare_data.py first.")
    raw = json.loads(path.read_text())
    return {k: [Sample(**d) for d in v] for k, v in raw.items()}


def load_gray(path: str | Path) -> np.ndarray:
    """Load an image as uint8 HxW."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
