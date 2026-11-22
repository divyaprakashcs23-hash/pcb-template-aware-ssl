"""Evaluation metrics.

Classification: accuracy, macro F1, per-class F1 and confusion matrix.

Detection: a small, dependency-free implementation so every number can be explained in the
viva.
* ``ap`` per class is the VOC all-point interpolated area under the precision-recall curve.
* ``mAP@0.5`` is the main metric. ``mAP@0.5:0.95`` averages over ten IoU thresholds (COCO).
* ``DeepPCB protocol``: the dataset paper counts a detection as correct at IoU > 0.33 and
  reports mAP and F-score at the best score threshold. Both are reproduced here.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_report(y_true, y_pred, num_classes: int) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(num_classes))
    per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "defect_macro_f1": float(np.mean(per[1:])),
        "per_class_f1": [float(v) for v in per],
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def _match_class(preds, gts, cls: int, iou_thr: float):
    """Greedy score-ordered matching for one class. Returns scores, tp flags and #gt."""
    scores, tps, n_gt = [], [], 0
    for p, g in zip(preds, gts):
        gb = g["boxes"][g["labels"] == cls]
        pm = p["labels"] == cls
        pb, ps = p["boxes"][pm], p["scores"][pm]
        n_gt += len(gb)
        if len(pb) == 0:
            continue
        order = np.argsort(-ps)
        pb, ps = pb[order], ps[order]
        iou = box_iou(pb, gb)
        used = np.zeros(len(gb), bool)
        for i in range(len(pb)):
            tp = False
            if len(gb):
                cand = np.where(~used & (iou[i] >= iou_thr))[0]
                if len(cand):
                    j = cand[np.argmax(iou[i, cand])]
                    used[j] = True
                    tp = True
            scores.append(ps[i])
            tps.append(tp)
    return np.asarray(scores, np.float32), np.asarray(tps, bool), n_gt


def _voc_ap(tp_sorted: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(tp_sorted) == 0:
        return 0.0
    ctp = np.cumsum(tp_sorted)
    cfp = np.cumsum(~tp_sorted)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def detection_ap(preds, gts, num_classes: int, iou_thr: float = 0.5) -> dict:
    """preds/gts: lists of dicts with numpy ``boxes`` (N,4), ``labels`` (N,), preds also ``scores``.
    Classes are 1..num_classes-1 (0 is background)."""
    aps, all_s, all_tp, total_gt = [], [], [], 0
    for c in range(1, num_classes):
        s, tp, n_gt = _match_class(preds, gts, c, iou_thr)
        order = np.argsort(-s)
        aps.append(_voc_ap(tp[order], n_gt))
        all_s.append(s)
        all_tp.append(tp)
        total_gt += n_gt
    s = np.concatenate(all_s) if all_s else np.zeros(0)
    tp = np.concatenate(all_tp) if all_tp else np.zeros(0, bool)
    order = np.argsort(-s)
    s, tp = s[order], tp[order]
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 1.0}
    if len(tp) and total_gt:
        ctp = np.cumsum(tp)
        prec = ctp / np.arange(1, len(tp) + 1)
        rec = ctp / total_gt
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        i = int(np.argmax(f1))
        best = {"f1": float(f1[i]), "precision": float(prec[i]), "recall": float(rec[i]),
                "threshold": float(s[i])}
    valid = [a for a in aps if not np.isnan(a)]
    return {"mAP": float(np.mean(valid)) if valid else 0.0, "ap_per_class": aps, "best_f": best}


def detection_report(preds, gts, num_classes: int) -> dict:
    r50 = detection_ap(preds, gts, num_classes, 0.5)
    r33 = detection_ap(preds, gts, num_classes, 0.33)
    coco = [detection_ap(preds, gts, num_classes, t)["mAP"] for t in np.arange(0.5, 0.96, 0.05)]
    return {
        "mAP50": r50["mAP"],
        "ap50_per_class": r50["ap_per_class"],
        "f1_at_best_50": r50["best_f"],
        "mAP50_95": float(np.mean(coco)),
        "deeppcb_mAP33": r33["mAP"],
        "deeppcb_F33": r33["best_f"],
        "n_images": len(gts),
    }
