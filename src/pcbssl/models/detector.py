"""Faster R-CNN with a ResNet-18 FPN backbone, initialised from any project encoder.

Anchor sizes are reduced to 16..256 because DeepPCB defects are small: the median box is
about 35 x 33 pixels and the 95th percentile is about 68 pixels on a 640 x 640 image.
Trainable BatchNorm is used for every initialisation so that randomly initialised backbones
can train at all and the comparison stays fair.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from ..config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES
from .encoder import load_encoder_state


def build_detector(init: str = "random", num_classes: int = NUM_CLASSES,
                   trainable_layers: int = 5) -> FasterRCNN:
    weights = ResNet18_Weights.IMAGENET1K_V1 if init == "imagenet" else None
    backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=weights,
                                   norm_layer=nn.BatchNorm2d, trainable_layers=trainable_layers)
    if init not in ("random", "imagenet"):
        state = load_encoder_state(init)
        missing, unexpected = backbone.body.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Backbone weights missing keys: {missing[:5]}")
    anchors = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)),
                              aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    return FasterRCNN(
        backbone, num_classes=num_classes, rpn_anchor_generator=anchors,
        min_size=IMAGE_SIZE, max_size=IMAGE_SIZE,
        image_mean=list(IMAGENET_MEAN), image_std=list(IMAGENET_STD),
        box_detections_per_img=100, box_score_thresh=0.01,
    )
