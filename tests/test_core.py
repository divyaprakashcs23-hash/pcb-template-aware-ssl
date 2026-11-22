"""Fast unit tests for the parts most likely to fail silently."""
import numpy as np
import pytest
import torch

from pcbssl.data.augment import GPUAugment, normalize
from pcbssl.data.deeppcb import Sample, parse_annotation
from pcbssl.data.sampler import label_fraction_subset
from pcbssl.engine.detect import dihedral
from pcbssl.eval.metrics import box_iou, detection_report
from pcbssl.models.simclr import nt_xent


def _samples(n=200):
    return [Sample(id=f"{i:05d}", group=f"g{i % 4}", test_path="", temp_path="") for i in range(n)]


def test_parse_annotation_space_and_comma(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("1 2 30 40 3\n5,6,50,60,6\n7 7 7 9 2\n")
    boxes, labels = parse_annotation(p)
    assert boxes == [[1, 2, 30, 40], [5, 6, 50, 60]]
    assert labels == [3, 6]


def test_label_fractions_nested_and_stratified():
    s = _samples()
    f10 = {x.id for x in label_fraction_subset(s, 0.10, seed=0)}
    f25 = {x.id for x in label_fraction_subset(s, 0.25, seed=0)}
    assert f10 <= f25
    assert len(f10) == 20 and len(f25) == 52  # ceil per group of 50
    assert {x.group for x in label_fraction_subset(s, 0.01, seed=0)} == {"g0", "g1", "g2", "g3"}
    assert f10 != {x.id for x in label_fraction_subset(s, 0.10, seed=1)}


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("flip", [False, True])
def test_dihedral_boxes_follow_pixels(k, flip):
    img = torch.zeros(64, 64)
    box = torch.tensor([[10.0, 20.0, 18.0, 40.0]])
    img[20:40, 10:18] = 1
    out, b = dihedral(img, box, k, flip)
    x1, y1, x2, y2 = b[0].long().tolist()
    assert out[y1:y2, x1:x2].sum() == img.sum()
    assert (x2 - x1) * (y2 - y1) == 8 * 20


def test_nt_xent_prefers_aligned_views():
    torch.manual_seed(0)
    z = torch.randn(32, 16)
    good, _ = nt_xent(z, z + 0.01 * torch.randn_like(z))
    bad, _ = nt_xent(z, torch.randn(32, 16))
    assert good < bad


def test_hard_negative_increases_loss_when_close():
    torch.manual_seed(0)
    z1 = torch.randn(16, 8)
    z2 = z1 + 0.1 * torch.randn_like(z1)
    mask = torch.ones(16, dtype=torch.bool)
    base, _ = nt_xent(z1, z2)
    near, st = nt_xent(z1, z2, hard_neg=z1 + 0.05 * torch.randn_like(z1), hard_mask=mask)
    far, _ = nt_xent(z1, z2, hard_neg=-z1, hard_mask=mask)
    assert near > far >= base - 1e-6
    assert "hard_gap" in st


def test_detection_metrics_perfect_and_empty():
    gts = [{"boxes": np.array([[0, 0, 10, 10], [20, 20, 40, 40]], np.float32), "labels": np.array([1, 2])}]
    perfect = [{"boxes": gts[0]["boxes"], "labels": gts[0]["labels"], "scores": np.array([0.9, 0.8])}]
    r = detection_report(perfect, gts, num_classes=7)
    assert r["mAP50"] == pytest.approx(1.0) and r["f1_at_best_50"]["f1"] == pytest.approx(1.0)
    empty = [{"boxes": np.zeros((0, 4)), "labels": np.zeros(0, int), "scores": np.zeros(0)}]
    assert detection_report(empty, gts, num_classes=7)["mAP50"] == 0.0


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], np.float32)
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)
    assert box_iou(a, np.array([[5, 0, 15, 10]], np.float32))[0, 0] == pytest.approx(1 / 3)


def test_gpu_augment_shapes_and_range():
    x = torch.rand(8, 1, 64, 64)
    for aug in (GPUAugment.ssl(), GPUAugment.finetune()):
        y = aug(x)
        assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    assert normalize(x).shape == (8, 3, 64, 64)
