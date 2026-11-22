"""Project-wide paths and constants."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PCBSSL_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DEEPPCB_ROOT = DATA_RAW / "DeepPCB" / "PCBData"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# Class 0 is background. DeepPCB defect IDs 1..6 keep their original meaning.
CLASS_NAMES = ["background", "open", "short", "mousebite", "spur", "copper", "pin-hole"]
NUM_CLASSES = len(CLASS_NAMES)
DEFECT_NAMES = CLASS_NAMES[1:]

PATCH_SIZE = 64
IMAGE_SIZE = 640

# All initialisations share one input normalisation so that the only difference between
# experiment arms is the encoder weights. Grayscale images are replicated to 3 channels.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LABEL_FRACTIONS = (0.10, 0.25, 1.00)
