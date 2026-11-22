# Label-Efficient PCB Defect Detection via Template-Aware Self-Supervised Pretraining

Deep Learning course project on the [DeepPCB](https://github.com/tangsanli5201/DeepPCB) dataset.

**Question.** Does contrastive pretraining on unlabelled PCB patches let a defect detector reach
good accuracy with a fraction of the labelled boxes? And does using the defect-free template,
which every automated optical inspection line already has, make that pretraining better?

**Headline deliverable.** One plot of detection mAP against label fraction, one line per
initialisation: `reports/figures/det_label_efficiency.png`. All numbers are in
[`reports/RESULTS.md`](reports/RESULTS.md), which is regenerated from the run folders.

## Method in one paragraph

A ResNet-18 encoder is pretrained with SimCLR on about 79k unlabelled 64 by 64 patches cut from the
training boards. The proposed **template-aware SimCLR** uses the aligned defect-free template in
two ways, guided by a label-free difference map between the tested image and the template.
Where the two agree, the template crop is an extra positive view. Where they differ, the template
crop is added as a **hard negative**, because the only thing separating it from the anchor is the
defect. A naive variant that always uses the template as a positive is kept as an ablation: it
asks the encoder to ignore defects, and it should hurt. Stage A fine-tunes each encoder as a
seven-class patch classifier. Stage B loads the same weights into a Faster R-CNN with a
ResNet-18 FPN backbone. Four initialisations are compared at several label fractions: random,
ImageNet, SimCLR and template-aware SimCLR.

## Repository layout

```
src/pcbssl/
  config.py              paths, class names, constants
  data/deeppcb.py        DeepPCB parser, official test split, stratified validation split
  data/sampler.py        seeded, nested, group-stratified label-fraction sampler
  data/patches.py        patch extraction, label-free difference maps, caches
  data/augment.py        batched GPU augmentations (per-sample random parameters)
  models/encoder.py      ResNet-18 with random / ImageNet / checkpoint init, patch classifier
  models/simclr.py       SimCLR, NT-Xent with per-anchor template hard negatives
  models/detector.py     Faster R-CNN ResNet-18 FPN with small anchors
  engine/pretrain.py     contrastive pretraining with kNN monitor and resume
  engine/classify.py     Stage A fine-tuning and linear probe
  engine/detect.py       Stage B detection training and evaluation
  eval/metrics.py        macro F1, VOC AP, mAP@0.5, mAP@0.5:0.95, DeepPCB F-score
  eval/gradcam.py        Grad-CAM
scripts/
  prepare_data.py        download DeepPCB, build splits and patch caches, dataset figures
  pretrain.py            one pretraining run
  train_classifier.py    one Stage A run
  train_detector.py      one Stage B run
  run_experiments.py     the full resumable grid
  make_report.py         tables, figures and reports/RESULTS.md
app/demo.py              Gradio demo: detection plus patch Grad-CAM
tests/test_core.py       unit tests for sampler, box transforms, loss and metrics
```

## Setup

Windows, Python 3.10 or newer, an NVIDIA GPU recommended (developed on an RTX 4070 Laptop, 8 GB).

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

## Reproduce everything

```bash
python scripts/prepare_data.py --download
```

```bash
python scripts/run_experiments.py
```

```bash
python scripts/make_report.py
```

```bash
python app/demo.py
```

`prepare_data.py` clones DeepPCB (about 220 MB) into `data/raw/` and writes caches to
`data/processed/`. `run_experiments.py` skips any run whose `metrics.json` already exists, so it can
be stopped and restarted. Use `--stage pretrain|cls|det|ablation|linear` to run one stage and
`--quick` for a five-minute smoke version of the whole pipeline.

```bash
python -m pytest -q
```

## Experimental protocol

| Item | Setting |
|---|---|
| Test set | Official DeepPCB test split, 500 images. 332 of them come from 4 board groups absent from training, reported separately as unseen designs. |
| Validation | 10% of trainval, stratified by board group, used only for model selection |
| Label fractions | Image-level, nested, stratified by group. Stage A: 2, 10, 25, 100%. Stage B: 10, 25, 100%. |
| Seeds | Stage A: 3 seeds. Stage B: 1 seed. |
| Pretraining | ResNet-18, 60 epochs, batch 512, AdamW 1e-3, cosine, temperature 0.2, fp16 |
| Stage A | 1500 steps regardless of fraction, batch 256, AdamW 5e-4, class-weighted CE, best validation macro F1 |
| Stage B | Faster R-CNN R18-FPN, anchors 16 to 256, 2500 iterations, batch 8, SGD 0.01, dihedral augmentation, best validation mAP@0.5 |
| Fairness | Same architecture, normalisation, augmentation, schedule and seeds for every initialisation. Only the starting weights change. |

## Syllabus coverage

| Syllabus topic | Where it appears |
|---|---|
| Convolutional networks | ResNet-18 encoder, `models/encoder.py` |
| Transfer learning and fine-tuning | ImageNet and self-supervised initialisations, full fine-tuning vs linear probe |
| Data augmentation | GPU random resized crop, dihedral group, blur, noise, `data/augment.py` |
| Self-supervised contrastive learning | SimCLR, NT-Xent, template-aware hard negatives, `models/simclr.py` |
| Object detection | Faster R-CNN, RPN, FPN, anchor design, `models/detector.py` |
| Class imbalance | 2:1 background ratio, inverse square-root class weights, macro F1 |
| Interpretability | Grad-CAM on layer3, `eval/gradcam.py`, t-SNE of embeddings |
| Evaluation methodology | Held-out test, validation-only model selection, seeds, nested label fractions, unseen-design split, mAP and F-score |

## Data licence

DeepPCB is released by its authors for research use only. Cite: Tang et al., *Online PCB Defect
Detector On A New PCB Defect Dataset*, arXiv:1902.06197.
