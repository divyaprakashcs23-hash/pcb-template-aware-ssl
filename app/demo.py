"""Gradio demo: PCB defect detection plus patch-level Grad-CAM.

    python app/demo.py                       # picks the best detector in runs/det
    python app/demo.py --detector runs/det/template_aware_f0.1_s0/model.pt

Tab 1 runs the Faster R-CNN on a whole 640x640 board image. If a defect-free template is also
given, the label-free difference map is shown next to the detections.
Tab 2 classifies a 64x64 patch and shows where the classifier looked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

from pcbssl.config import CLASS_NAMES, NUM_CLASSES, PATCH_SIZE, RUNS_DIR
from pcbssl.data.augment import normalize, to_input
from pcbssl.data.deeppcb import load_splits
from pcbssl.data.patches import difference_maps
from pcbssl.eval.gradcam import GradCAM
from pcbssl.models.detector import build_detector
from pcbssl.models.encoder import PatchClassifier, build_resnet18

COLORS = ["#999999", "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def best_run(pattern: str, key) -> Path | None:
    best, best_v = None, -1
    for f in RUNS_DIR.glob(pattern):
        if not (f.parent / "model.pt").exists():
            continue
        v = key(json.loads(f.read_text()))
        if v > best_v:
            best, best_v = f.parent / "model.pt", v
    return best


def load_detector(path: Path):
    m = build_detector("random")
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return m.to(DEV).eval()


def load_classifier(path: Path):
    m = PatchClassifier(build_resnet18("random"), NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return m.to(DEV).eval()


def to_gray(img) -> np.ndarray:
    if img is None:
        return None
    return np.asarray(Image.fromarray(np.asarray(img)).convert("L"), dtype=np.uint8)


def make_detect_fn(model):
    @torch.no_grad()
    def detect(test_img, template_img, threshold):
        g = to_gray(test_img)
        if g is None:
            return None, None, "Upload a tested PCB image."
        x = torch.as_tensor(g).float().div(255).unsqueeze(0).expand(3, -1, -1).to(DEV)
        o = model([x])[0]
        vis = Image.fromarray(g).convert("RGB")
        draw = ImageDraw.Draw(vis)
        rows = []
        for b, l, s in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy()):
            if s < threshold:
                continue
            draw.rectangle(b.tolist(), outline=COLORS[l], width=3)
            draw.text((b[0] + 2, max(0, b[1] - 12)), f"{CLASS_NAMES[l]} {s:.2f}", fill=COLORS[l])
            rows.append(f"| {CLASS_NAMES[l]} | {s:.2f} | {int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])} |")
        diff_vis = None
        t = to_gray(template_img)
        if t is not None and t.shape == g.shape:
            loose, strict = difference_maps(g, t)
            d = np.full((*g.shape, 3), 255, np.uint8)
            d[loose] = (160, 160, 160)
            d[strict] = (220, 30, 30)
            diff_vis = Image.fromarray(d)
        summary = f"**{len(rows)} defects** at score at least {threshold:.2f}\n\n"
        if rows:
            summary += "| class | score | box x1,y1,x2,y2 |\n|---|---|---|\n" + "\n".join(rows)
        return vis, diff_vis, summary
    return detect


def make_classify_fn(model):
    def classify(patch):
        g = to_gray(patch)
        if g is None:
            return None, {}
        g = np.asarray(Image.fromarray(g).resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR))
        x = normalize(to_input(torch.as_tensor(g)[None].to(DEV)))
        with torch.no_grad():
            prob = model(x).softmax(1)[0].cpu().numpy()
        cam = GradCAM(model, model.encoder.layer3)
        maps, _ = cam(x)
        cam.remove()
        heat = (maps[0].cpu().numpy() * 255).astype(np.uint8)
        import matplotlib.cm as cm
        rgb = (cm.jet(heat)[..., :3] * 255).astype(np.uint8)
        over = (0.55 * np.stack([g] * 3, -1) + 0.45 * rgb).astype(np.uint8)
        over = Image.fromarray(over).resize((256, 256), Image.NEAREST)
        return over, {CLASS_NAMES[i]: float(p) for i, p in enumerate(prob)}
    return classify


def examples(n: int = 4):
    try:
        test = load_splits()["test"]
    except FileNotFoundError:
        return []
    step = max(1, len(test) // n)
    return [[s.test_path, s.temp_path] for s in test[::step][:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default=None)
    ap.add_argument("--classifier", default=None)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    det_path = Path(a.detector) if a.detector else best_run("det/*/metrics.json", lambda m: m["test"]["mAP50"])
    cls_path = Path(a.classifier) if a.classifier else best_run(
        "cls/finetune/*/metrics.json", lambda m: m["test"]["macro_f1"])
    if det_path is None:
        sys.exit("No trained detector found. Run scripts/run_experiments.py --stage det first.")

    with gr.Blocks(title="PCB defect inspector") as ui:
        gr.Markdown("# PCB defect inspector\nFaster R-CNN (ResNet-18 FPN) trained on DeepPCB. "
                    f"Detector: `{det_path.parent.name}`" + (f", classifier: `{cls_path.parent.name}`" if cls_path else ""))
        with gr.Tab("Detect defects"):
            with gr.Row():
                test_in = gr.Image(label="Tested board (640x640)", type="numpy")
                temp_in = gr.Image(label="Defect-free template (optional)", type="numpy")
            thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Score threshold")
            btn = gr.Button("Detect", variant="primary")
            with gr.Row():
                out_img = gr.Image(label="Detections")
                out_diff = gr.Image(label="Label-free difference map (red = strong)")
            out_md = gr.Markdown()
            btn.click(make_detect_fn(load_detector(det_path)), [test_in, temp_in, thr], [out_img, out_diff, out_md])
            ex = examples()
            if ex:
                gr.Examples(ex, [test_in, temp_in])
        if cls_path:
            with gr.Tab("Explain a patch (Grad-CAM)"):
                p_in = gr.Image(label="Patch (resized to 64x64)", type="numpy")
                p_btn = gr.Button("Classify", variant="primary")
                with gr.Row():
                    p_cam = gr.Image(label="Grad-CAM")
                    p_lab = gr.Label(num_top_classes=4, label="Prediction")
                p_btn.click(make_classify_fn(load_classifier(cls_path)), [p_in], [p_cam, p_lab])
    ui.launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
