"""Seeded, nested, group-stratified label-fraction sampling at the image level.

Labelling cost is paid per board image, so fractions are taken over images, not patches.
For a fixed seed the subsets are nested: the 10% subset is contained in the 25% subset,
which is contained in 100%. That makes curves across fractions directly comparable.
"""
from __future__ import annotations

import math

import numpy as np

from .deeppcb import Sample


def label_fraction_subset(samples: list[Sample], fraction: float, seed: int) -> list[Sample]:
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction >= 1:
        return list(samples)
    rng = np.random.default_rng(10_000 + seed)
    by_group: dict[str, list[Sample]] = {}
    for s in samples:
        by_group.setdefault(s.group, []).append(s)
    chosen = []
    for g in sorted(by_group):
        items = sorted(by_group[g], key=lambda s: s.id)
        perm = rng.permutation(len(items))  # independent of fraction, so subsets are nested
        k = max(1, math.ceil(fraction * len(items)))
        chosen += [items[i] for i in perm[:k]]
    return sorted(chosen, key=lambda s: s.id)
