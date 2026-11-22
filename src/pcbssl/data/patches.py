"""Patch extraction and caching.

Two patch sets are built:

* **SSL patches** (unlabelled). Dense grid crops of every training board plus crops centred on
  connected components of the test-vs-template difference map. Both use no annotations,
  so pretraining never sees a label. Each patch is stored with the template crop at the
  same location and a label-free *difference state*:
    0 = clean      (test and template agree; template crop is a valid positive view)
    1 = differs    (strong difference; template crop is a hard negative)
    2 = ambiguous  (only thin edge noise; template is ignored)
* **Classification patches** (labelled, 7 classes). One patch per annotated defect, centred on
  its box, plus background patches that do not touch any defect box.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm

from ..config import PATCH_SIZE
from .deeppcb import Sample, load_gray

CLEAN, DIFFERS, AMBIGUOUS = 0, 1, 2


def difference_maps(test: np.ndarray, temp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label-free defect evidence from the aligned template.

    Returns a high-recall map (3x3 opening) and a high-precision map (5x5 opening).
    On DeepPCB trainval the 3x3 map touches about 99.5% of annotated defects and the 5x5
    map localises defects with about 97% precision.
    """
    d = (test > 127) ^ (temp > 127)
    loose = ndi.binary_opening(d, structure=np.ones((3, 3), bool))
    strict = ndi.binary_opening(d, structure=np.ones((5, 5), bool))
    return loose, strict


def diff_components(strict: np.ndarray, min_area: int = 10) -> list[tuple[int, int]]:
    """Centres (cy, cx) of strict-difference components, used to propose defect-like crops."""
    lab, n = ndi.label(strict)
    if n == 0:
        return []
    areas = ndi.sum(strict, lab, range(1, n + 1))
    centres = []
    for i, sl in enumerate(ndi.find_objects(lab)):
        if sl is not None and areas[i] >= min_area:
            centres.append(((sl[0].start + sl[0].stop) // 2, (sl[1].start + sl[1].stop) // 2))
    return centres


def _crop(img: np.ndarray, cy: int, cx: int, size: int, out: int = PATCH_SIZE) -> np.ndarray:
    """Crop a size x size window centred at (cy, cx) with reflect padding, resized to out."""
    half = size // 2
    pad = size
    padded = np.pad(img, pad, mode="reflect")
    y0, x0 = cy - half + pad, cx - half + pad
    patch = padded[y0:y0 + size, x0:x0 + size]
    if size != out:
        patch = np.asarray(Image.fromarray(patch).resize((out, out), Image.BILINEAR))
    return patch


def _state(loose: np.ndarray, strict: np.ndarray, y0: int, x0: int, size: int, min_px: int = 10) -> int:
    s = strict[y0:y0 + size, x0:x0 + size].sum()
    if s >= min_px:
        return DIFFERS
    if loose[y0:y0 + size, x0:x0 + size].sum() >= min_px:
        return AMBIGUOUS
    return CLEAN


def build_ssl_patches(samples: list[Sample], seed: int = 0, size: int = PATCH_SIZE,
                      diff_jitter: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    tests, temps, states, img_idx, kinds = [], [], [], [], []
    for i, s in enumerate(tqdm(samples, desc="ssl patches")):
        test, temp = load_gray(s.test_path), load_gray(s.temp_path)
        H, W = test.shape
        loose, strict = difference_maps(test, temp)
        # 1) dense grid with a random offset per image
        oy, ox = rng.integers(0, size, size=2)
        for y0 in range(int(oy) % size, H - size + 1, size):
            for x0 in range(int(ox) % size, W - size + 1, size):
                tests.append(test[y0:y0 + size, x0:x0 + size])
                temps.append(temp[y0:y0 + size, x0:x0 + size])
                states.append(_state(loose, strict, y0, x0, size))
                img_idx.append(i)
                kinds.append(0)
        # 2) crops centred on label-free difference components
        for cy, cx in diff_components(strict):
            cy = int(np.clip(cy + rng.integers(-diff_jitter, diff_jitter + 1), size // 2, H - size // 2))
            cx = int(np.clip(cx + rng.integers(-diff_jitter, diff_jitter + 1), size // 2, W - size // 2))
            y0, x0 = cy - size // 2, cx - size // 2
            tests.append(test[y0:y0 + size, x0:x0 + size])
            temps.append(temp[y0:y0 + size, x0:x0 + size])
            states.append(_state(loose, strict, y0, x0, size))
            img_idx.append(i)
            kinds.append(1)
    return {
        "test": np.stack(tests).astype(np.uint8),
        "temp": np.stack(temps).astype(np.uint8),
        "state": np.asarray(states, np.uint8),
        "img_idx": np.asarray(img_idx, np.int32),
        "kind": np.asarray(kinds, np.uint8),
        "ids": np.asarray([s.id for s in samples]),
    }


def _overlaps(y0, x0, size, boxes, margin=4) -> bool:
    for x1, y1, x2, y2 in boxes:
        if x0 < x2 + margin and x0 + size > x1 - margin and y0 < y2 + margin and y0 + size > y1 - margin:
            return True
    return False


def build_cls_patches(samples: list[Sample], seed: int = 0, bg_ratio: float = 2.0,
                      size: int = PATCH_SIZE, context: float = 1.2) -> dict[str, np.ndarray]:
    """Labelled 7-class patches. Large defects are cropped with context and resized to 64."""
    rng = np.random.default_rng(seed)
    out_p, out_t, out_y, out_img, out_box = [], [], [], [], []
    for i, s in enumerate(tqdm(samples, desc="cls patches")):
        test, temp = load_gray(s.test_path), load_gray(s.temp_path)
        H, W = test.shape
        for (x1, y1, x2, y2), c in zip(s.boxes, s.labels):
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            crop = max(size, int(np.ceil(context * max(x2 - x1, y2 - y1))))
            crop += crop % 2
            out_p.append(_crop(test, cy, cx, crop, size))
            out_t.append(_crop(temp, cy, cx, crop, size))
            out_y.append(c)
            out_img.append(i)
            out_box.append([x1, y1, x2, y2])
        n_bg = int(round(bg_ratio * max(1, len(s.boxes))))
        tries = 0
        while n_bg > 0 and tries < 200:
            tries += 1
            y0, x0 = rng.integers(0, H - size), rng.integers(0, W - size)
            if _overlaps(y0, x0, size, s.boxes):
                continue
            out_p.append(test[y0:y0 + size, x0:x0 + size])
            out_t.append(temp[y0:y0 + size, x0:x0 + size])
            out_y.append(0)
            out_img.append(i)
            out_box.append([x0, y0, x0 + size, y0 + size])
            n_bg -= 1
    return {
        "x": np.stack(out_p).astype(np.uint8),
        "temp": np.stack(out_t).astype(np.uint8),
        "y": np.asarray(out_y, np.int64),
        "img_idx": np.asarray(out_img, np.int32),
        "box": np.asarray(out_box, np.int32),
        "ids": np.asarray([s.id for s in samples]),
        "groups": np.asarray([s.group for s in samples]),
    }


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/prepare_data.py first.")
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
