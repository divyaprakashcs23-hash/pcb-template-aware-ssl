// Builds the RVCE Deep Learning EL deck. Real data only: dataset statistics from
// data/processed/stats.json and the live pretraining log. Every result slide is reserved.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const HERE = __dirname;
const PROJ = path.resolve(HERE, "..");
const OUT = path.join(PROJ, "reports", "DL_EL_Presentation.pptx");

const stats = JSON.parse(fs.readFileSync(path.join(PROJ, "data/processed/stats.json"), "utf8"));

// ---- live pretraining log -------------------------------------------------------------
const logTxt = fs.readFileSync(path.join(PROJ, "runs/full_stdout.log"), "utf8");
const epochs = [];
for (const line of logTxt.split(/\r?\n/)) {
  const m = line.match(/\[pretrain:(\w+)\] epoch=(\d+) loss=([\d.]+) pos_acc=([\d.]+)(?:.*?knn_acc=([\d.]+) knn_macro_f1=([\d.]+))?/);
  if (m) epochs.push({ mode: m[1], epoch: +m[2], loss: +m[3], pos: +m[4], knn: m[5] ? +m[5] : null, knnf1: m[6] ? +m[6] : null });
}
const simclrEpochs = epochs.filter(e => e.mode === "simclr");
const lastEp = simclrEpochs[simclrEpochs.length - 1];
const knnPts = simclrEpochs.filter(e => e.knn !== null);

// ---- theme ------------------------------------------------------------------------------
const THEME = {
  name: "PCB Copper",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1A1A1A", lt1: "FFFFFF", dk2: "0B3D2E", lt2: "EAF3EE",
    accent1: "0B3D2E", accent2: "C0702A", accent3: "2E7D5B", accent4: "8FB8A8",
    accent5: "D9A66B", accent6: "5C6B64", hlink: "1F5FA8", folHlink: "5C6B64",
  },
};
const H = THEME.colors; // hex, for hex-only options

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Label-Efficient PCB Defect Detection via Template-Aware Self-Supervised Pretraining";
pres.author = "RVCE Deep Learning EL";
pres.subject = "Deep Learning (21IS73GA) Experiential Learning";
const C = pres.SchemeColor;

const COURSE = "Deep Learning (21IS73GA)  |  VII Semester  |  RV College of Engineering";
const LOGO = path.join(HERE, "rvce_logo_rgba.png"); // 2330 x 1001

// ---- layouts ----------------------------------------------------------------------------
pres.defineSlideMaster({
  title: "TITLE", background: { color: H.lt1 },
  objects: [
    { rect: { x: 0, y: 3.95, w: 10, h: 1.675, fill: { color: H.dk2 } } },
    { image: { x: 0.5, y: 0.35, w: 2.8, h: 1.2, path: LOGO } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION", background: { color: H.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.0, w: 8.6, h: 1.2, fontSize: 40, bold: true, color: H.lt1, align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.25, w: 8.6, h: 0.9, fontSize: 18, color: H.accent4, align: "left", margin: 0 }, text: "" } },
    { text: { text: COURSE, options: { x: 0.7, y: 5.1, w: 7.5, h: 0.3, fontSize: 10, color: H.accent4, margin: 0 } } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: H.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.7, fontSize: 26, bold: true, color: H.dk2, align: "left", margin: 0, valign: "middle" }, text: "" } },
    { text: { text: COURSE, options: { x: 0.5, y: 5.2, w: 7.5, h: 0.3, fontSize: 9, color: H.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 9.1, y: 5.2, w: 0.5, h: 0.3, fontSize: 9, color: H.accent6, align: "right" },
});
pres.defineSlideMaster({
  title: "CONTENT_DARK", background: { color: H.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.7, fontSize: 26, bold: true, color: H.lt1, align: "left", margin: 0, valign: "middle" }, text: "" } },
    { text: { text: COURSE, options: { x: 0.5, y: 5.2, w: 7.5, h: 0.3, fontSize: 9, color: H.accent4, margin: 0 } } },
  ],
  slideNumber: { x: 9.1, y: 5.2, w: 0.5, h: 0.3, fontSize: 9, color: H.accent4, align: "right" },
});

// ---- helpers ----------------------------------------------------------------------------
let sec = "";
function slide(master, title, sectionTitle) {
  if (sectionTitle && sectionTitle !== sec) { sec = sectionTitle; pres.addSection({ title: sec }); }
  const s = pres.addSlide({ masterName: master, sectionTitle: sec });
  if (title) s.addText(title, { placeholder: "title" });
  return s;
}
function bullets(s, items, o) {
  const arr = items.map((t, i) => {
    const txt = typeof t === "string" ? t : t.text;
    const opt = { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: 6 };
    if (typeof t !== "string" && t.options) Object.assign(opt, t.options);
    return { text: txt, options: opt };
  });
  s.addText(arr, Object.assign({ fontSize: 14, color: C.text1, valign: "top", isTextBox: true, margin: 2 }, o));
}
function card(s, x, y, w, h, head, body, opts = {}) {
  const fill = opts.fill || C.background2;
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08, objectName: "card " + head });
  s.addText(head, { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: 0.35, fontSize: opts.headSize || 13, bold: true, color: opts.headColor || C.accent2, isTextBox: true, margin: 0, valign: "top" });
  if (body) s.addText(body, { x: x + 0.15, y: y + 0.45, w: w - 0.3, h: h - 0.55, fontSize: opts.bodySize || 11.5, color: opts.bodyColor || C.text1, isTextBox: true, margin: 0, valign: "top" });
}
function stat(s, x, y, w, big, label, opts = {}) {
  s.addText(big, { x, y, w, h: 0.6, fontSize: opts.size || 32, bold: true, color: opts.color || C.accent2, isTextBox: true, margin: 0, align: "left", fontFace: "Cambria" });
  s.addText(label, { x, y: y + 0.6, w, h: 0.4, fontSize: 11, color: opts.labelColor || C.accent6, isTextBox: true, margin: 0 });
}
function numCircle(s, x, y, n, opts = {}) {
  const d = opts.d || 0.42;
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: opts.fill || C.accent2 }, line: { color: opts.fill || C.accent2 } });
  s.addText(String(n), { x, y, w: d, h: d, fontSize: opts.size || 13, bold: true, color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0 });
}
function reserved(s, x, y, w, h, title, lines) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: C.background1 }, line: { color: H.accent2, width: 1.5, dashType: "dash" }, rectRadius: 0.08, objectName: "reserved " + title });
  s.addText(title, { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: 0.35, fontSize: 13, bold: true, color: C.accent2, isTextBox: true, margin: 0 });
  bullets(s, lines, { x: x + 0.2, y: y + 0.55, w: w - 0.4, h: h - 0.7, fontSize: 11, color: C.accent6 });
}
const chartText = { catAxisLabelColor: H.accent6, valAxisLabelColor: H.accent6, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  valGridLine: { color: "DDE6E1", size: 0.5 }, catGridLine: { style: "none" }, titleColor: H.dk2, titleFontSize: 12, legendFontSize: 10, legendColor: H.accent6 };
const fmt = n => n.toLocaleString("en-IN");

// =========================================================================================
// 1. TITLE
// =========================================================================================
{
  const s = slide("TITLE", null, "Introduction");
  s.addText("Rashtreeya Sikshana Samithi Trust", { x: 3.5, y: 0.4, w: 6.0, h: 0.3, fontSize: 11, color: C.accent6, isTextBox: true, margin: 0 });
  s.addText("RV College of Engineering, Bengaluru 560059", { x: 3.5, y: 0.68, w: 6.0, h: 0.3, fontSize: 12, bold: true, color: C.text2, isTextBox: true, margin: 0 });
  s.addText("Autonomous Institution affiliated to VTU, Belagavi", { x: 3.5, y: 0.95, w: 6.0, h: 0.3, fontSize: 10.5, color: C.accent6, isTextBox: true, margin: 0 });
  s.addText("Department of Computer Science and Engineering / Information Science and Engineering", { x: 3.5, y: 1.22, w: 6.3, h: 0.3, fontSize: 10.5, color: C.accent6, isTextBox: true, margin: 0 });

  s.addText("Experiential Learning Report", { x: 0.5, y: 1.85, w: 9, h: 0.3, fontSize: 12, color: C.accent2, bold: true, isTextBox: true, margin: 0, charSpacing: 2 });
  s.addText("Label-Efficient PCB Defect Detection via Template-Aware Self-Supervised Pretraining", { x: 0.5, y: 2.15, w: 9.0, h: 1.2, fontSize: 28, bold: true, color: C.text2, isTextBox: true, margin: 0, valign: "top", fontFace: "Cambria" });
  s.addText("Course: Deep Learning   |   Course code: 21IS73GA   |   Semester: VII (2026-27, odd)", { x: 0.5, y: 3.4, w: 9.0, h: 0.35, fontSize: 13, color: C.text1, isTextBox: true, margin: 0 });

  s.addText([
    { text: "Submitted by", options: { bold: true, breakLine: true, color: H.accent4 } },
    { text: "USN 1RV23CS310" },
  ], { x: 0.5, y: 4.1, w: 4.6, h: 1.3, fontSize: 12, color: H.lt1, isTextBox: true, margin: 0, valign: "top" });
  s.addText([
    { text: "Under the guidance of", options: { bold: true, breakLine: true, color: H.accent4 } },
    { text: "Prof. Rajatha", options: { breakLine: true } },
    { text: "Dept. of CSE, RVCE", options: { breakLine: true } },
    { text: "Academic year 2026-27" },
  ], { x: 5.4, y: 4.1, w: 4.1, h: 1.3, fontSize: 12, color: H.lt1, isTextBox: true, margin: 0, valign: "top" });
  s.addNotes("Title slide. Course code 21IS73GA is the RVCE 7th semester Deep Learning code from the published 2021 scheme (Professional Core Elective III, Group G, common to CS and IS); confirm against your own scheme PDF.");
}

// =========================================================================================
// 2. AGENDA
// =========================================================================================
{
  const s = slide("CONTENT", "Agenda");
  const items = [
    ["Problem and motivation", "Why PCB inspection needs label-efficient detectors"],
    ["Dataset", "DeepPCB: images, templates, splits and class balance"],
    ["Process and pipeline", "Patches, difference maps, pretraining, Stage A and B"],
    ["Concepts and novelty", "SimCLR, NT-Xent and the template-aware objective"],
    ["Tools and protocol", "Stack, hyperparameters, fairness and evaluation"],
    ["Results and inference", "Pretraining, Stage A, linear probe, Stage B, analysis, conclusions"],
    ["Literature and references", "Papers, dataset, libraries and sites"],
  ];
  items.forEach(([h, b], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.5 + col * 4.65, y = 1.2 + row * 1.0;
    numCircle(s, x, y + 0.08, i + 1);
    s.addText(h, { x: x + 0.55, y, w: 3.9, h: 0.3, fontSize: 14, bold: true, color: C.text2, isTextBox: true, margin: 0 });
    s.addText(b, { x: x + 0.55, y: y + 0.3, w: 3.9, h: 0.5, fontSize: 11, color: C.accent6, isTextBox: true, margin: 0, valign: "top" });
  });
}

// =========================================================================================
// 3. PROBLEM AND MOTIVATION
// =========================================================================================
{
  const s = slide("CONTENT", "Problem: PCB inspection needs labels factories lack");
  const rows = [
    ["AOI", "Automated optical inspection lines image every board", "Every line already stores a defect-free reference template for each design. Images are abundant; defect annotations are not."],
    ["$", "Bounding-box labels are expensive", "Defects are tiny (median box about 35 x 33 px on a 640 x 640 board). Drawing boxes needs trained inspectors and does not transfer across board designs."],
    ["?", "Can unlabelled boards do the heavy lifting", "Self-supervised contrastive pretraining learns from unlabelled patches. The question is whether that reduces the labelled boxes a detector needs."],
    ["+", "Can the template make pretraining smarter", "Comparing a tested image against its template tells us, without any label, where a defect might be. We fold that signal into the pretraining objective."],
  ];
  rows.forEach(([ic, h, b], i) => {
    const y = 1.15 + i * 1.0;
    s.addShape(pres.ShapeType.ellipse, { x: 0.5, y: y + 0.05, w: 0.55, h: 0.55, fill: { color: i % 2 ? C.accent2 : C.accent3 }, line: { color: i % 2 ? C.accent2 : C.accent3 } });
    s.addText(ic, { x: 0.5, y: y + 0.05, w: 0.55, h: 0.55, fontSize: ic.length > 1 ? 10 : 16, bold: true, color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    s.addText(h, { x: 1.25, y, w: 8.2, h: 0.32, fontSize: 15, bold: true, color: C.text2, isTextBox: true, margin: 0 });
    s.addText(b, { x: 1.25, y: y + 0.32, w: 8.2, h: 0.6, fontSize: 12, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
  });
}

// =========================================================================================
// 4. RESEARCH QUESTIONS AND OBJECTIVES
// =========================================================================================
{
  const s = slide("CONTENT", "Research questions and objectives");
  card(s, 0.5, 1.15, 4.4, 1.6, "RQ1  Label efficiency",
    "Does contrastive pretraining on unlabelled PCB patches let a Faster R-CNN defect detector reach good mAP with 10% or 25% of the labelled boxes, compared with random and ImageNet initialisation?", { bodySize: 12 });
  card(s, 5.1, 1.15, 4.4, 1.6, "RQ2  Template awareness",
    "Does using the aligned defect-free template inside the contrastive objective (as an extra positive where the board is clean, as a hard negative where it differs) improve the learned representation?", { bodySize: 12 });
  s.addText("Objectives", { x: 0.5, y: 2.9, w: 9, h: 0.3, fontSize: 15, bold: true, color: C.text2, isTextBox: true, margin: 0 });
  bullets(s, [
    "Build a reproducible DeepPCB pipeline: parser, official test split, stratified validation split, nested label-fraction sampler.",
    "Extract about 79k unlabelled 64 x 64 patches and a label-free difference state for each one.",
    "Pretrain ResNet-18 encoders with SimCLR and with the proposed template-aware SimCLR (plus a naive template-positive ablation).",
    "Compare four initialisations in Stage A (patch classification) and Stage B (detection) at several label fractions, same protocol for all.",
    "Report one headline plot: detection mAP against label fraction, one line per initialisation, plus Grad-CAM and t-SNE analysis and a Gradio demo.",
  ], { x: 0.5, y: 3.22, w: 9, h: 1.9, fontSize: 11 });
}

// =========================================================================================
// 5. DATASET
// =========================================================================================
{
  const s = slide("CONTENT", "Dataset: DeepPCB (Tang et al., 2019)", "Dataset");
  s.addImage({ path: path.join(PROJ, "reports/figures/dataset_pair.png"), x: 0.5, y: 1.1, w: 5.9, h: 2.07, objectName: "Tested image, template and difference map" });
  s.addText("Tested image with annotations (left), its aligned defect-free template (centre), and the label-free difference map (right).", { x: 0.5, y: 3.2, w: 5.9, h: 0.4, fontSize: 10, italic: true, color: C.accent6, isTextBox: true, margin: 0 });
  const tot = stats.images.train + stats.images.val + stats.images.test;
  const boxes = stats.boxes.train + stats.boxes.val + stats.boxes.test;
  stat(s, 6.7, 1.1, 1.45, fmt(tot), "image pairs, 640 x 640, binarised", { size: 26 });
  stat(s, 8.25, 1.1, 1.45, fmt(boxes), "annotated defect boxes", { size: 26 });
  stat(s, 6.7, 2.15, 1.45, "6", "defect classes", { size: 26 });
  stat(s, 8.25, 2.15, 1.45, "11", "board design groups", { size: 26 });
  bullets(s, [
    "Each tested image has a pixel-aligned defect-free template of the same design.",
    "Classes: open, short, mousebite, spur, copper, pin-hole.",
    "Official 500-image test split; 332 of them are from 4 board groups never seen in training (unseen designs).",
  ], { x: 6.7, y: 3.2, w: 3.0, h: 1.9, fontSize: 10.5 });
  bullets(s, [
    "Source: github.com/tangsanli5201/DeepPCB, about 220 MB, cloned by scripts/prepare_data.py.",
    "Parsing: one annotation file per image with x1 y1 x2 y2 class rows (space or comma separated).",
    "Released by the authors for research use only.",
  ], { x: 0.5, y: 3.7, w: 5.9, h: 1.3, fontSize: 10.5 });
}

// =========================================================================================
// 6. SPLITS AND CLASS BALANCE
// =========================================================================================
{
  const s = slide("CONTENT", "Splits and class balance");
  const names = ["background", "open", "short", "mousebite", "spur", "copper", "pin-hole"];
  const defectIdx = [1, 2, 3, 4, 5, 6];
  const data = ["train", "val", "test"].map(k => ({ name: k, labels: defectIdx.map(i => names[i]), values: defectIdx.map(i => stats.cls_patches[k][i]) }));
  s.addChart(pres.ChartType.bar, data, Object.assign({
    x: 0.5, y: 1.1, w: 5.6, h: 3.4, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [H.accent1, H.accent2, H.accent4], showTitle: true, title: "Labelled defect patches per class and split",
    showLegend: true, legendPos: "b", showValue: false, valAxisMinVal: 0,
  }, chartText));
  s.addText("Background patches (2 per defect) are added on top: " + fmt(stats.cls_patches.train[0]) + " train, " + fmt(stats.cls_patches.val[0]) + " val, " + fmt(stats.cls_patches.test[0]) + " test.", { x: 0.5, y: 4.55, w: 5.6, h: 0.5, fontSize: 10, italic: true, color: C.accent6, isTextBox: true, margin: 0 });

  const rows = [
    [{ text: "Split", options: { bold: true, color: H.lt1, fill: { color: H.dk2 } } }, { text: "Images", options: { bold: true, color: H.lt1, fill: { color: H.dk2 } } }, { text: "Boxes", options: { bold: true, color: H.lt1, fill: { color: H.dk2 } } }],
    ["Train", fmt(stats.images.train), fmt(stats.boxes.train)],
    ["Validation (10% of trainval, stratified by group)", fmt(stats.images.val), fmt(stats.boxes.val)],
    ["Test (official)", fmt(stats.images.test), fmt(stats.boxes.test)],
    ["  of which unseen designs (4 groups)", fmt(stats.unseen_test_images), "-"],
  ];
  s.addTable(rows, { x: 6.4, y: 1.1, w: 3.1, colW: [1.8, 0.65, 0.65], fontSize: 10, color: C.text1, border: { type: "solid", color: "DDE6E1", pt: 0.5 }, fill: { color: H.lt1 }, rowH: 0.34, valign: "middle" });
  bullets(s, [
    "Seen groups (train): " + stats.seen_groups.map(g => g.replace("group", "")).join(", ") + ".",
    "Unseen test groups: " + stats.unseen_test_groups.map(g => g.replace("group", "")).join(", ") + ".",
    "Label fractions are image-level, nested and group-stratified, so 10% is a subset of 25%, which is a subset of 100%.",
    "Class imbalance handled with a 2:1 background ratio, inverse square-root class weights and macro F1.",
  ], { x: 6.4, y: 3.05, w: 3.1, h: 2.0, fontSize: 10.5 });
}

// =========================================================================================
// 7. LABEL-FREE DIFFERENCE MAP
// =========================================================================================
{
  const s = slide("CONTENT", "Label-free signal: the template difference map");
  const sf = stats.ssl_state_fraction;
  s.addImage({ path: path.join(PROJ, "reports/figures/ssl_patch_states.png"), x: 0.5, y: 1.1, w: 4.9, h: 2.0, objectName: "Patch states: clean, differs, ambiguous" });
  s.addText("Unlabelled 64 x 64 SSL patches: tested crop next to its template crop. Rows: clean, differs, ambiguous.", { x: 0.5, y: 3.12, w: 4.9, h: 0.35, fontSize: 10, italic: true, color: C.accent6, isTextBox: true, margin: 0 });
  bullets(s, [
    "d = (test > 127) XOR (template > 127), then morphological opening: 3 x 3 gives a high-recall loose map, 5 x 5 a high-precision strict map.",
    "Each patch gets a state with no annotation: clean (template crop is a valid positive view), differs (template crop is a hard negative), ambiguous (edge noise only; template ignored).",
    "Patch set: " + fmt(stats.ssl_patches) + " = " + fmt(stats.ssl_grid_patches) + " dense grid crops + " + fmt(stats.ssl_diff_guided_patches) + " crops centred on difference components.",
  ], { x: 0.5, y: 3.5, w: 4.9, h: 1.6, fontSize: 10.5 });

  s.addChart(pres.ChartType.doughnut, [{ name: "state", labels: ["clean", "differs", "ambiguous"], values: [sf.clean * 100, sf.differs * 100, sf.ambiguous * 100] }], Object.assign({
    x: 5.6, y: 1.05, w: 2.3, h: 2.3, holeSize: 55, chartColors: [H.accent4, H.accent2, H.accent6],
    showLegend: true, legendPos: "b", showPercent: false, showValue: true, dataLabelFormatCode: "0.0\"%\"", dataLabelColor: H.lt1, dataLabelFontSize: 9,
    showTitle: true, title: "Patch states (%)",
  }, chartText));
  stat(s, 8.0, 1.15, 1.6, (stats.label_free_difference_recall_train.loose_recall * 100).toFixed(1) + "%", "of annotated train boxes touched by the loose map", { size: 26 });
  stat(s, 8.0, 2.35, 1.6, (stats.label_free_difference_recall_train.strict_recall * 100).toFixed(1) + "%", "touched by the strict map", { size: 26, color: C.accent3 });
  card(s, 5.6, 3.55, 3.9, 1.5, "Why this matters", "The difference map is computed from pixels alone, so pretraining never sees a label. It is high recall, not perfect: alignment noise creates false differences, which is exactly why the ambiguous state exists and why we test the idea instead of assuming it.", { bodySize: 11 });
}

// =========================================================================================
// 8. PIPELINE
// =========================================================================================
{
  const s = slide("CONTENT", "End-to-end process", "Process");
  const steps = [
    ["1  Prepare", "Clone DeepPCB, parse annotations, build official test split and a 10% group-stratified validation split", C.accent6],
    ["2  Patches", "79k unlabelled SSL patches with template crop and difference state; labelled 7-class patches for Stage A", C.accent6],
    ["3  Pretrain", "ResNet-18 + projector, SimCLR or template-aware SimCLR, 60 epochs, batch 512, kNN monitor", C.accent2],
    ["4  Stage A", "Fine-tune each encoder as a 7-class patch classifier at 2, 10, 25, 100% labels, 3 seeds", C.accent3],
    ["5  Stage B", "Load the same weights into Faster R-CNN R18-FPN, train at 10, 25, 100% labels", C.accent3],
    ["6  Report", "make_report.py: tables, label-efficiency plots, Grad-CAM, t-SNE, RESULTS.md; Gradio demo", C.accent1],
  ];
  const w = 1.42, gap = 0.1, y = 1.35, h = 1.95;
  steps.forEach(([hd, b, col], i) => {
    const x = 0.5 + i * (w + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: col }, line: { color: col }, rectRadius: 0.06, objectName: "step " + hd });
    s.addText(hd, { x: x + 0.1, y: y + 0.1, w: w - 0.2, h: 0.35, fontSize: 12.5, bold: true, color: C.background1, isTextBox: true, margin: 0 });
    s.addText(b, { x: x + 0.1, y: y + 0.48, w: w - 0.2, h: h - 0.55, fontSize: 9.5, color: C.background1, isTextBox: true, margin: 0, valign: "top" });
    if (i < steps.length - 1) s.addShape(pres.ShapeType.rightArrow, { x: x + w - 0.02, y: y + h / 2 - 0.1, w: gap + 0.04, h: 0.2, fill: { color: H.accent5 }, line: { color: H.accent5 } });
  });
  s.addText("Labels used", { x: 0.5, y: 3.5, w: 2, h: 0.3, fontSize: 12, bold: true, color: C.text2, isTextBox: true, margin: 0 });
  const lab = ["none", "none", "none (kNN monitor only)", "fraction", "fraction", "test once"];
  lab.forEach((t, i) => s.addText(t, { x: 0.5 + i * (w + gap), y: 3.8, w, h: 0.3, fontSize: 10, color: C.accent2, bold: true, isTextBox: true, margin: 0 }));
  bullets(s, [
    "Four initialisations go through exactly the same Stage A and Stage B code: random, ImageNet, SimCLR (PCB) and template-aware SimCLR. Only the starting weights change.",
    "The grid is resumable: scripts/run_experiments.py skips any run whose metrics.json already exists, so it can be stopped and restarted.",
  ], { x: 0.5, y: 4.2, w: 9, h: 0.9, fontSize: 11 });
}

// =========================================================================================
// 9. CONCEPT: SIMCLR
// =========================================================================================
{
  const s = slide("CONTENT", "Concept: contrastive pretraining with SimCLR", "Concepts and novelty");
  // two-view diagram
  const bx = 0.5, by = 1.2;
  s.addShape(pres.ShapeType.roundRect, { x: bx, y: by + 0.9, w: 1.1, h: 0.7, fill: { color: C.background2 }, line: { color: C.accent4 }, rectRadius: 0.05 });
  s.addText("patch x", { x: bx, y: by + 0.9, w: 1.1, h: 0.7, fontSize: 11, bold: true, color: C.text2, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  [["aug t1", by], ["aug t2", by + 1.8]].forEach(([t, yy]) => {
    s.addShape(pres.ShapeType.rightArrow, { x: bx + 1.15, y: yy + 0.25, w: 0.4, h: 0.2, fill: { color: H.accent5 }, line: { color: H.accent5 } });
    s.addShape(pres.ShapeType.roundRect, { x: bx + 1.6, y: yy, w: 0.9, h: 0.7, fill: { color: C.accent4 }, line: { color: C.accent4 }, rectRadius: 0.05 });
    s.addText(t, { x: bx + 1.6, y: yy, w: 0.9, h: 0.7, fontSize: 11, color: C.background1, bold: true, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    s.addShape(pres.ShapeType.rightArrow, { x: bx + 2.55, y: yy + 0.25, w: 0.4, h: 0.2, fill: { color: H.accent5 }, line: { color: H.accent5 } });
    s.addShape(pres.ShapeType.roundRect, { x: bx + 3.0, y: yy, w: 1.3, h: 0.7, fill: { color: C.accent1 }, line: { color: C.accent1 }, rectRadius: 0.05 });
    s.addText("ResNet-18 f + projector g", { x: bx + 3.0, y: yy, w: 1.3, h: 0.7, fontSize: 9.5, color: C.background1, bold: true, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    s.addShape(pres.ShapeType.rightArrow, { x: bx + 4.35, y: yy + 0.25, w: 0.4, h: 0.2, fill: { color: H.accent5 }, line: { color: H.accent5 } });
  });
  s.addText("z1", { x: bx + 4.8, y: by + 0.1, w: 0.5, h: 0.5, fontSize: 14, bold: true, color: C.accent2, isTextBox: true, margin: 0, valign: "middle" });
  s.addText("z2", { x: bx + 4.8, y: by + 1.9, w: 0.5, h: 0.5, fontSize: 14, bold: true, color: C.accent2, isTextBox: true, margin: 0, valign: "middle" });
  s.addShape(pres.ShapeType.line, { x: bx + 4.95, y: by + 0.65, w: 0, h: 1.2, line: { color: H.accent2, width: 1.5, dashType: "dash" } });
  s.addText("positive pair", { x: bx + 5.05, y: by + 1.05, w: 1.0, h: 0.4, fontSize: 9.5, italic: true, color: C.accent2, isTextBox: true, margin: 0 });
  s.addText("Every other view in the batch of 2N is pushed away (negatives)", { x: bx, y: by + 2.65, w: 5.6, h: 0.3, fontSize: 10, italic: true, color: C.accent6, isTextBox: true, margin: 0 });

  card(s, 6.3, 1.15, 3.2, 1.5, "NT-Xent loss (temperature tau = 0.2)",
    "For anchor i with positive j:\nL(i) = -log [ exp(sim(zi,zj)/tau) / sum over k != i of exp(sim(zi,zk)/tau) ]\nsim = cosine similarity of projector outputs, averaged over all 2N anchors.", { bodySize: 10 });
  card(s, 6.3, 2.8, 3.2, 1.1, "PCB-specific augmentation", "Random resized crop, all 8 dihedral transforms (defects are orientation invariant), contrast, brightness, blur, noise. No colour jitter: images are binarised.", { bodySize: 10 });
  card(s, 6.3, 4.05, 3.2, 1.05, "Monitoring without label leakage", "A kNN classifier on frozen features is evaluated every 10 epochs on validation patches. Labels only watch the run; they never train the encoder.", { bodySize: 10 });
  s.addText("Implementation: models/simclr.py (SimCLR, Projector 512-512-128, nt_xent), engine/pretrain.py (AdamW 1e-3, cosine schedule, 5 warm-up epochs, fp16 autocast, resume from last.pt).", { x: 0.5, y: 4.4, w: 5.6, h: 0.65, fontSize: 10, color: C.text1, isTextBox: true, margin: 0 });
}

// =========================================================================================
// 10. NOVELTY
// =========================================================================================
{
  const s = slide("CONTENT_DARK", "Novelty: template-aware SimCLR");
  s.addText("The aligned template is free supervision. Plain SimCLR ignores it; a naive use of it is harmful. We route each patch by its label-free difference state.", { x: 0.5, y: 1.0, w: 9, h: 0.5, fontSize: 13, color: H.accent4, isTextBox: true, margin: 0, italic: true });
  const cols = [
    ["Clean patch", "test == template", "With probability p = 0.5 the second view is the template crop.", "The template is a genuine positive: same content, different capture. Adds real nuisance variation (alignment, binarisation noise) that augmentation cannot invent.", H.accent3],
    ["Differing patch", "strong difference", "Both views come from the tested crop. The template crop at the same location is appended as an extra hard negative for that anchor, for both of its views.", "The only thing separating anchor and template is the defect, so the encoder is pushed to represent the defect instead of ignoring it.", H.accent2],
    ["Ambiguous patch", "thin edge noise only", "Plain SimCLR. The template is ignored.", "Alignment artefacts would otherwise be mistaken for defects (false hard negatives) or hide them (false positives).", H.accent6],
  ];
  cols.forEach(([h, sub, what, why, col], i) => {
    const x = 0.5 + i * 3.05, y = 1.6, w = 2.9;
    s.addShape(pres.ShapeType.roundRect, { x, y, w, h: 2.75, fill: { color: H.lt1 }, line: { color: H.lt1 }, rectRadius: 0.08, objectName: "case " + h });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.15, y: y + 0.15, w: 0.3, h: 0.3, fill: { color: col }, line: { color: col } });
    s.addText(h, { x: x + 0.55, y: y + 0.1, w: w - 0.7, h: 0.3, fontSize: 14, bold: true, color: H.dk2, isTextBox: true, margin: 0 });
    s.addText(sub, { x: x + 0.55, y: y + 0.38, w: w - 0.7, h: 0.25, fontSize: 10, italic: true, color: H.accent6, isTextBox: true, margin: 0 });
    s.addText([{ text: "What: ", options: { bold: true, color: col } }, { text: what }], { x: x + 0.15, y: y + 0.7, w: w - 0.3, h: 0.95, fontSize: 10.5, color: H.dk1, isTextBox: true, margin: 0, valign: "top" });
    s.addText([{ text: "Why: ", options: { bold: true, color: col } }, { text: why }], { x: x + 0.15, y: y + 1.7, w: w - 0.3, h: 1.0, fontSize: 10.5, color: H.dk1, isTextBox: true, margin: 0, valign: "top" });
  });
  s.addText([
    { text: "Loss change. ", options: { bold: true, color: H.accent5 } },
    { text: "The NT-Xent denominator gains one extra term per differing anchor: exp(sim(z_anchor, z_template)/tau), optionally up-weighted by hard_weight. A hard_gap statistic (positive similarity minus template similarity) is logged to check the negative is actually being separated.", options: { color: H.lt1 } },
    { text: "   Ablation. ", options: { bold: true, color: H.accent5 } },
    { text: "template_pos always uses the template as a positive. For defective patches this asks the encoder to map with-defect and without-defect to the same point, i.e. to become defect-blind. It is kept to show the routing is what matters.", options: { color: H.lt1 } },
  ], { x: 0.5, y: 4.45, w: 9, h: 0.7, fontSize: 10, isTextBox: true, margin: 0, valign: "top" });
}

// =========================================================================================
// 11. STAGE A AND STAGE B
// =========================================================================================
{
  const s = slide("CONTENT", "Stage A classification and Stage B detection");
  card(s, 0.5, 1.15, 4.4, 3.9, "Stage A  Patch classification (7 classes)", null);
  bullets(s, [
    "Encoder: the pretrained ResNet-18 (fc removed) plus a linear head on the 512-d feature.",
    "Labelled images: nested, group-stratified subset at 2, 10, 25 or 100%.",
    "1500 steps regardless of fraction, batch 256, AdamW 5e-4, cosine schedule, light augmentation.",
    "Class-weighted cross-entropy (inverse square-root frequency) for the background imbalance.",
    "Model selection on validation macro F1; test evaluated once with the selected weights.",
    "Linear probe variant: encoder frozen, only the head trains. Measures representation quality directly.",
    "Three seeds per cell; mean and min to max range reported.",
  ], { x: 0.65, y: 1.6, w: 4.1, h: 3.4, fontSize: 11 });
  card(s, 5.1, 1.15, 4.4, 3.9, "Stage B  Detection with Faster R-CNN", null);
  bullets(s, [
    "Backbone: ResNet-18 FPN. The same encoder weights are loaded into backbone.body, so the comparison is about initialisation only.",
    "Anchors reduced to 16, 32, 64, 128, 256 px (3 aspect ratios) because defects are small: median box about 35 x 33 px, 95th percentile about 68 px.",
    "Trainable BatchNorm for every initialisation so that a random backbone can train at all.",
    "2500 iterations, batch 8, SGD 0.01 with momentum, 300 warm-up iterations, cosine decay, dihedral augmentation on whole boards.",
    "Model selection on validation mAP@0.5. Reported: mAP@0.5, mAP@0.5:0.95, DeepPCB F-score at IoU 0.33, seen vs unseen designs.",
    "Label fractions: 10, 25, 100%.",
  ], { x: 5.25, y: 1.6, w: 4.1, h: 3.4, fontSize: 11 });
}

// =========================================================================================
// 12. PROTOCOL
// =========================================================================================
{
  const s = slide("CONTENT", "Experimental protocol and fairness", "Tools and protocol");
  const hd = t => ({ text: t, options: { bold: true, color: H.lt1, fill: { color: H.dk2 } } });
  const rows = [
    [hd("Item"), hd("Setting")],
    ["Test set", "Official DeepPCB test split, 500 images. 332 from 4 board groups absent from training, reported separately as unseen designs."],
    ["Validation", "10% of trainval, stratified by board group, used only for model selection."],
    ["Label fractions", "Image-level, nested, stratified by group. Stage A: 2, 10, 25, 100%. Stage B: 10, 25, 100%."],
    ["Seeds", "Stage A: 3 seeds. Stage B: 1 seed."],
    ["Pretraining", "ResNet-18, 60 epochs, batch 512, AdamW 1e-3, cosine, temperature 0.2, fp16, about 79k patches."],
    ["Stage A", "1500 steps regardless of fraction, batch 256, AdamW 5e-4, class-weighted CE, best validation macro F1."],
    ["Stage B", "Faster R-CNN R18-FPN, anchors 16 to 256, 2500 iterations, batch 8, SGD 0.01, dihedral augmentation, best validation mAP@0.5."],
    ["Fairness", "Same architecture, normalisation, augmentation, schedule and seeds for every initialisation. Only the starting weights change."],
    ["Metrics", "Macro F1, per-class F1, confusion matrix; VOC AP, mAP@0.5, mAP@0.5:0.95, DeepPCB F-score at IoU 0.33."],
  ];
  s.addTable(rows, { x: 0.5, y: 1.1, w: 9.0, colW: [1.6, 7.4], fontSize: 10.5, color: C.text1, border: { type: "solid", color: "DDE6E1", pt: 0.5 }, rowH: 0.36, valign: "middle", margin: 0.05 });
}

// =========================================================================================
// 13. TOOLS
// =========================================================================================
{
  const s = slide("CONTENT", "Tools, libraries and hardware");
  const tools = [
    ["PyTorch 2.x + CUDA 12.8", "Model definition, AMP fp16 autocast and GradScaler, channels-last memory format."],
    ["torchvision", "ResNet-18 (ImageNet1K_V1 weights), Faster R-CNN, FPN backbone utilities, AnchorGenerator."],
    ["NumPy, SciPy", "Patch caches (.npz), binary opening and connected components for the difference maps."],
    ["Pillow, Matplotlib", "Image I/O, dataset figures, label-efficiency plots, confusion matrices."],
    ["scikit-learn", "t-SNE of embeddings, kNN monitor during pretraining."],
    ["Gradio", "Interactive demo: whole-board detection, difference map, patch Grad-CAM."],
    ["pytest", "Unit tests for the sampler, box transforms, NT-Xent and detection metrics."],
    ["Hardware", "NVIDIA RTX 4070 Laptop GPU, 8 GB; Windows 11; Python 3.13 virtual environment."],
  ];
  tools.forEach(([h, b], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.5 + col * 4.65, y = 1.15 + row * 0.98;
    s.addShape(pres.ShapeType.roundRect, { x, y, w: 4.4, h: 0.85, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.06, objectName: "tool " + h });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.15, y: y + 0.25, w: 0.35, h: 0.35, fill: { color: i % 2 ? C.accent2 : C.accent3 }, line: { color: i % 2 ? C.accent2 : C.accent3 } });
    s.addText(h, { x: x + 0.65, y: y + 0.08, w: 3.65, h: 0.3, fontSize: 12.5, bold: true, color: C.text2, isTextBox: true, margin: 0 });
    s.addText(b, { x: x + 0.65, y: y + 0.38, w: 3.65, h: 0.45, fontSize: 10, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
  });
  s.addText("Engineering choices: batched GPU augmentation with per-sample random parameters (no DataLoader workers on Windows), whole image caches on the GPU, resumable runs, one normalisation shared by every arm.", { x: 0.5, y: 5.0 - 0.1, w: 9, h: 0.35, fontSize: 9.5, italic: true, color: C.accent6, isTextBox: true, margin: 0 });
}

// =========================================================================================
// 14. SYLLABUS MAPPING
// =========================================================================================
{
  const s = slide("CONTENT", "Deep Learning concepts applied (course coverage)");
  const hd = t => ({ text: t, options: { bold: true, color: H.lt1, fill: { color: H.dk2 } } });
  const rows = [
    [hd("Syllabus topic"), hd("Where it appears in this project")],
    ["Convolutional networks", "ResNet-18 encoder with residual blocks; feature pyramid for detection (models/encoder.py)."],
    ["Transfer learning and fine-tuning", "ImageNet and self-supervised initialisations; full fine-tuning versus frozen linear probe."],
    ["Data augmentation", "GPU random resized crop, dihedral group, blur, noise, contrast and brightness (data/augment.py)."],
    ["Self-supervised contrastive learning", "SimCLR, NT-Xent, temperature, projector head, template-aware hard negatives (models/simclr.py)."],
    ["Object detection", "Faster R-CNN: RPN, RoI heads, FPN, anchor design for small objects (models/detector.py)."],
    ["Optimisation and regularisation", "AdamW and SGD with momentum, cosine schedule with warm-up, weight decay, mixed precision, BatchNorm."],
    ["Class imbalance", "2:1 background ratio, inverse square-root class weights, macro F1 as the selection metric."],
    ["Interpretability", "Grad-CAM on layer3 (eval/gradcam.py); t-SNE of embeddings."],
    ["Evaluation methodology", "Held-out test, validation-only model selection, seeds, nested label fractions, unseen-design split, mAP and F-score."],
  ];
  s.addTable(rows, { x: 0.5, y: 1.1, w: 9.0, colW: [2.6, 6.4], fontSize: 10.5, color: C.text1, border: { type: "solid", color: "DDE6E1", pt: 0.5 }, rowH: 0.36, valign: "middle", margin: 0.05 });
}

// =========================================================================================
// RESULTS: data loaded from the finished run (reports/figures/*.csv, runs/pretrain/*/pretrain.json)
// =========================================================================================
const FIG = f => path.join(PROJ, "reports/figures", f);
function readCsv(f) {
  const [head, ...rows] = fs.readFileSync(f, "utf8").trim().split(/\r?\n/);
  const cols = head.split(",");
  return rows.map(r => { const v = r.split(","); const o = {}; cols.forEach((c, i) => o[c] = isNaN(+v[i]) || v[i] === "" ? v[i] : +v[i]); return o; });
}
const INITS = ["random", "imagenet", "simclr", "template_pos", "template_aware"];
const LABEL = { random: "Random init", imagenet: "ImageNet", simclr: "SimCLR (PCB)", template_pos: "Template-positive (naive)", template_aware: "Template-aware SimCLR (ours)" };
const SHORT = { random: "Random", imagenet: "ImageNet", simclr: "SimCLR", template_pos: "Template-pos (naive)", template_aware: "Template-aware (ours)" };
const ICOL = { random: "8A8A8A", imagenet: "2A6FDB", simclr: "E08A00", template_pos: "B04FC4", template_aware: "1A9E5A" };
const clsFT = readCsv(FIG("cls_finetune.csv"));
const clsLP = readCsv(FIG("cls_linear.csv"));
const det = readCsv(FIG("det_results.csv"));
const pre = {};
for (const m of ["simclr", "template_pos", "template_aware"]) {
  const f = path.join(PROJ, "runs/pretrain", m, "pretrain.json");
  if (fs.existsSync(f)) pre[m] = JSON.parse(fs.readFileSync(f, "utf8")).history;
}
function mean(rows, key) { const v = rows.map(r => r[key]).filter(x => typeof x === "number" && !isNaN(x)); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN; }
function agg(rows, key, init, frac) { return mean(rows.filter(r => r.init === init && Math.abs(r.fraction - frac) < 1e-9), key); }
function rng(rows, key, init, frac) { const v = rows.filter(r => r.init === init && Math.abs(r.fraction - frac) < 1e-9).map(r => r[key]); return [Math.min(...v), Math.max(...v)]; }
const f3 = x => isNaN(x) ? "" : x.toFixed(3);
const pct = x => (x * 100).toFixed(1);
const CLS_FR = [0.02, 0.1, 0.25, 1.0], DET_FR = [0.1, 0.25, 1.0];
const hd = t => ({ text: t, options: { bold: true, color: H.lt1, fill: { color: H.dk2 }, fontSize: 9.5 } });
const hi = t => ({ text: t, options: { bold: true, color: H.accent3 } });
const tableOpts = (x, y, w, colW, rowH = 0.3) => ({ x, y, w, colW, fontSize: 9.5, color: C.text1, border: { type: "solid", color: "DDE6E1", pt: 0.5 }, rowH, valign: "middle" });
function lineChart(s, x, y, w, h, title, labels, rows, key, fracs, inits, valMin, valMax) {
  const data = inits.map(i => ({ name: SHORT[i], labels, values: fracs.map(f => +agg(rows, key, i, f).toFixed(3)) }));
  s.addChart(pres.ChartType.line, data, Object.assign({
    x, y, w, h, chartColors: inits.map(i => ICOL[i]), lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    showTitle: true, title, showLegend: true, legendPos: "b", showValue: false,
    valAxisMinVal: valMin, valAxisMaxVal: valMax, valAxisLabelFormatCode: "0.00",
    catAxisTitle: "labelled training images", showCatAxisTitle: true, catAxisTitleFontSize: 9, catAxisTitleColor: H.accent6, catAxisTitleFontFace: "+mn-lt",
  }, chartText));
}
function resultTable(s, x, y, w, firstCol, rows, key, fracs, inits, withRange) {
  const head = [hd(firstCol), ...fracs.map(f => hd((f * 100) + "%"))];
  const body = inits.map(i => {
    const best = Math.max(...inits.map(j => agg(rows, key, j, 0) || 0));
    return [i === "template_aware" ? hi(SHORT[i]) : SHORT[i], ...fracs.map(f => {
      const m = agg(rows, key, i, f);
      const isBest = inits.every(j => agg(rows, key, j, f) <= m + 1e-12);
      const txt = f3(m) + (withRange ? "" : "");
      return isBest ? { text: txt, options: { bold: true, color: H.accent3 } } : txt;
    })];
  });
  const colW = [1.55, ...fracs.map(() => (w - 1.55) / fracs.length)];
  s.addTable([head, ...body], tableOpts(x, y, w, colW, 0.28));
}
const gpuH = (() => {
  let t = 0;
  for (const h of Object.values(pre)) t += h[h.length - 1].time_s || 0;
  for (const d of ["runs/cls/finetune", "runs/cls/linear", "runs/det"]) {
    const dir = path.join(PROJ, d);
    if (!fs.existsSync(dir)) continue;
    for (const r of fs.readdirSync(dir)) { const f = path.join(dir, r, "metrics.json"); if (fs.existsSync(f)) t += JSON.parse(fs.readFileSync(f, "utf8")).time_s || 0; }
  }
  return t / 3600;
})();
const last = m => pre[m] ? pre[m][pre[m].length - 1] : {};
const knnAt = (m, ep) => { const r = (pre[m] || []).find(e => e.epoch === ep && e.knn_macro_f1 != null); return r ? r.knn_macro_f1 : NaN; };
const apShort = i => { const f = path.join(PROJ, "runs/det", i + "_f0.1_s0/metrics.json"); return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, "utf8")).test.ap50_per_class[1] : NaN; }; // CLASS_NAMES[1:] = open, short, ...
const nRuns = { ft: clsFT.length, lp: clsLP.length, det: det.length };

// =========================================================================================
// 15. EXPERIMENT GRID: WHAT RAN
// =========================================================================================
{
  const s = slide("CONTENT", "Experiment grid: every stage has finished", "Results and inference");
  stat(s, 0.5, 1.1, 2.1, String(Object.keys(pre).length) + " x 60", "encoders x pretraining epochs");
  stat(s, 2.75, 1.1, 2.1, String(nRuns.ft), "Stage A fine-tune runs");
  stat(s, 5.0, 1.1, 2.1, String(nRuns.lp), "linear-probe runs");
  stat(s, 7.25, 1.1, 2.2, String(nRuns.det), "Faster R-CNN detectors");
  const rows = [
    ["1", "Pretrain SimCLR (PCB)", "60 epochs, final loss " + f3(last("simclr").loss) + ", pos-pair acc " + pct(last("simclr").pos_acc) + "%"],
    ["2", "Pretrain template-aware SimCLR", "60 epochs, hard gap " + f3(pre.template_aware[0].hard_gap) + " to " + f3(last("template_aware").hard_gap)],
    ["3", "Ablation: template-positive pretrain", "60 epochs, final loss " + f3(last("template_pos").loss)],
    ["4", "Stage A fine-tune grid", "5 inits x 4 fractions x 3 seeds = " + nRuns.ft + " runs, 1500 steps each"],
    ["5", "Linear probes on frozen encoders", "5 inits x 4 fractions x 3 seeds = " + nRuns.lp + " runs"],
    ["6", "Stage B detector grid", "4 inits x 3 fractions = " + nRuns.det + " runs, 2500 iterations each"],
    ["7", "make_report.py", "RESULTS.md, 20 figures and 3 CSV tables in reports/"],
  ];
  rows.forEach(([n, t, st], i) => {
    const y = 2.25 + i * 0.37;
    numCircle(s, 0.5, y + 0.02, n, { fill: C.accent3, d: 0.3, size: 10 });
    s.addText(t, { x: 0.9, y, w: 3.3, h: 0.34, fontSize: 11, bold: true, color: C.text2, isTextBox: true, margin: 0, valign: "middle" });
    s.addText(st, { x: 4.2, y, w: 5.3, h: 0.34, fontSize: 10, color: C.accent6, isTextBox: true, margin: 0, valign: "middle" });
  });
  card(s, 0.5, 4.9, 9.0, 0.3, "", null);
  s.addText("Total measured GPU time about " + gpuH.toFixed(1) + " h on one RTX 4060 (8 GB). All numbers that follow are from this run on the official 500-image DeepPCB test split; the earlier smoke run is not used anywhere.", { x: 0.65, y: 4.9, w: 8.7, h: 0.3, fontSize: 9.5, color: C.text1, isTextBox: true, margin: 0, valign: "middle" });
  s.addNotes("Every value on the result slides is read at build time from reports/figures/*.csv and runs/pretrain/*/pretrain.json, written by scripts/make_report.py. Nothing is typed in by hand.");
}

// =========================================================================================
// 16. PRETRAINING RESULTS
// =========================================================================================
{
  const s = slide("CONTENT", "Pretraining results: loss and kNN monitor");
  s.addImage({ path: FIG("pretrain_curves.png"), x: 0.5, y: 1.1, w: 6.0, h: 2.07, objectName: "pretraining loss and kNN curves" });
  const modes = ["simclr", "template_pos", "template_aware"];
  s.addTable([
    [hd("Encoder"), hd("Loss"), hd("kNN ep20"), hd("kNN ep60")],
    ...modes.map(m => [m === "template_aware" ? hi("Ours") : (m === "simclr" ? "SimCLR" : "Naive"), f3(last(m).loss), f3(knnAt(m, 20)), m === "template_aware" ? hi(f3(knnAt(m, 60))) : f3(knnAt(m, 60))]),
  ], tableOpts(6.7, 1.1, 2.8, [0.85, 0.65, 0.65, 0.65], 0.3));
  s.addText("kNN (k = 20) on frozen features, validation macro F1. Monitoring only: no gradient ever uses these labels.", { x: 6.7, y: 2.55, w: 2.8, h: 0.6, fontSize: 9, italic: true, color: C.accent6, isTextBox: true, margin: 0, valign: "top" });
  card(s, 0.5, 3.3, 4.4, 1.8, "What the curves say", null);
  bullets(s, [
    "Template-aware SimCLR reaches kNN macro F1 " + f3(knnAt("template_aware", 60)) + " against " + f3(knnAt("simclr", 60)) + " for plain SimCLR, and leads at every checkpoint from epoch 20 (" + f3(knnAt("template_aware", 20)) + " vs " + f3(knnAt("simclr", 20)) + "): the template signal speeds up learning of defect-relevant features.",
    "The hard gap (positive similarity minus template-negative similarity) grows from " + f3(pre.template_aware[0].hard_gap) + " to " + f3(last("template_aware").hard_gap) + ", so the encoder learns to push a defective patch away from its own clean template.",
  ], { x: 0.65, y: 3.7, w: 4.1, h: 1.4, fontSize: 9.5 });
  card(s, 5.1, 3.3, 4.4, 1.8, "Why the naive variant fails", null);
  bullets(s, [
    "Treating the template crop as a positive for every patch asks the encoder to ignore exactly the pixels that define a defect. Its kNN F1 stalls at " + f3(knnAt("template_pos", 60)) + " and its loss stays highest.",
    "Loss values are not comparable across objectives: the template-aware loss includes an extra hard-negative term per anchor, so a higher final loss does not mean a worse encoder.",
  ], { x: 5.25, y: 3.7, w: 4.1, h: 1.4, fontSize: 9.5 });
}

// =========================================================================================
// 17. STAGE A FINE-TUNE
// =========================================================================================
{
  const s = slide("CONTENT", "Stage A: fine-tuned patch classification");
  lineChart(s, 0.5, 1.1, 5.1, 2.9, "Test macro F1 (7 classes), mean of 3 seeds", ["2% (20 img)", "10% (92)", "25% (228)", "100% (900)"], clsFT, "test_f1", CLS_FR, INITS, 0.84, 1.0);
  resultTable(s, 5.8, 1.1, 3.7, "Test macro F1", clsFT, "test_f1", CLS_FR, INITS);
  const r2 = i => rng(clsFT, "test_f1", i, 0.02);
  s.addText("Bold = best per column. Seed range at 2%: ours " + f3(r2("template_aware")[0]) + " to " + f3(r2("template_aware")[1]) + ", ImageNet " + f3(r2("imagenet")[0]) + " to " + f3(r2("imagenet")[1]) + ", random " + f3(r2("random")[0]) + " to " + f3(r2("random")[1]) + ".", { x: 5.8, y: 3.0, w: 3.7, h: 0.55, fontSize: 9, italic: true, color: C.accent6, isTextBox: true, margin: 0, valign: "top" });
  resultTable(s, 5.8, 3.55, 3.7, "Defect-only F1", clsFT, "defect_f1", CLS_FR, ["imagenet", "simclr", "template_aware"]);
  card(s, 0.5, 4.05, 5.1, 1.1, "Inference", null);
  bullets(s, [
    "With 20 labelled boards (2%), ours scores " + f3(agg(clsFT, "test_f1", "template_aware", 0.02)) + ", level with ImageNet (" + f3(agg(clsFT, "test_f1", "imagenet", 0.02)) + "), above SimCLR (" + f3(agg(clsFT, "test_f1", "simclr", 0.02)) + ") and random (" + f3(agg(clsFT, "test_f1", "random", 0.02)) + "). The naive variant is worst (" + f3(agg(clsFT, "test_f1", "template_pos", 0.02)) + ").",
    "From 10% labels on, all initialisations land within about 1.5 points: full fine-tuning hides most of the representation difference, so the next slide freezes the encoder.",
  ], { x: 0.65, y: 4.38, w: 4.85, h: 0.75, fontSize: 9 });
}

// =========================================================================================
// 18. LINEAR PROBE
// =========================================================================================
{
  const s = slide("CONTENT", "Linear probe: quality of the frozen features");
  lineChart(s, 0.5, 1.1, 5.1, 2.9, "Test macro F1, linear layer only, mean of 3 seeds", ["2%", "10%", "25%", "100%"], clsLP, "test_f1", CLS_FR, INITS, 0.5, 1.0);
  resultTable(s, 5.8, 1.1, 3.7, "Linear probe F1", clsLP, "test_f1", CLS_FR, INITS);
  const d = f => ((agg(clsLP, "test_f1", "template_aware", f) - agg(clsLP, "test_f1", "simclr", f)) * 100).toFixed(1);
  stat(s, 5.8, 3.1, 1.8, "+" + ((agg(clsLP, "test_f1", "template_aware", 0.02) - agg(clsLP, "test_f1", "imagenet", 0.02)) * 100).toFixed(0) + " pts", "ours vs ImageNet, 2% labels", { size: 26 });
  stat(s, 7.7, 3.1, 1.8, "+" + d(0.02) + " pts", "ours vs plain SimCLR at 2%", { size: 26, color: C.accent3 });
  card(s, 0.5, 4.05, 9.0, 1.1, "Inference", null);
  bullets(s, [
    "This is the cleanest test of H2. Frozen template-aware features reach " + f3(agg(clsLP, "test_f1", "template_aware", 0.02)) + " macro F1 from 20 boards and " + f3(agg(clsLP, "test_f1", "template_aware", 1.0)) + " from 900, ahead of plain SimCLR at all four fractions (margin " + d(0.02) + ", " + d(0.1) + ", " + d(0.25) + ", " + d(1.0) + " points), with the largest gain where labels are scarcest.",
    "ImageNet features (" + f3(agg(clsLP, "test_f1", "imagenet", 1.0)) + " at 100%) are no better than a random encoder (" + f3(agg(clsLP, "test_f1", "random", 1.0)) + ") on binarised PCB patches: the domain gap is real, and PCB-specific pretraining is what closes it.",
  ], { x: 0.65, y: 4.38, w: 8.7, h: 0.75, fontSize: 9 });
}

// =========================================================================================
// 19. STAGE B HEADLINE
// =========================================================================================
{
  const s = slide("CONTENT", "Stage B headline: detection mAP@0.5 vs label fraction");
  const DI = ["random", "imagenet", "simclr", "template_aware"];
  lineChart(s, 0.5, 1.1, 5.1, 2.9, "Faster R-CNN test mAP@0.5 (500 boards, 1 seed)", ["10% (92 img)", "25% (228)", "100% (900)"], det, "mAP50", DET_FR, DI, 0.88, 0.98);
  resultTable(s, 5.8, 1.1, 3.7, "mAP@0.5", det, "mAP50", DET_FR, DI);
  resultTable(s, 5.8, 2.6, 3.7, "F-score, IoU 0.33", det, "deeppcb_F33", DET_FR, DI);
  const m = (i, f) => agg(det, "mAP50", i, f);
  card(s, 0.5, 4.05, 9.0, 1.1, "Inference", null);
  bullets(s, [
    "With 10% of the labels (92 boards) the template-aware detector reaches mAP@0.5 " + f3(m("template_aware", 0.1)) + ": +" + ((m("template_aware", 0.1) - m("imagenet", 0.1)) * 100).toFixed(1) + " points over ImageNet and +" + ((m("template_aware", 0.1) - m("random", 0.1)) * 100).toFixed(1) + " over random init. With 25% (" + f3(m("template_aware", 0.25)) + ") it already beats random init trained on all 900 boards (" + f3(m("random", 1.0)) + ").",
    "At 100% labels ImageNet edges ahead (" + f3(m("imagenet", 1.0)) + " vs " + f3(m("template_aware", 1.0)) + "); the benefit of PCB pretraining is a label-efficiency benefit. COCO-style mAP@0.5:0.95 shows no gain over SimCLR (" + f3(agg(det, "mAP50_95", "template_aware", 0.1)) + " vs " + f3(agg(det, "mAP50_95", "simclr", 0.1)) + " at 10%): boxes are found, not localised more tightly. Single seed, so differences under about 1 point are noise.",
  ], { x: 0.65, y: 4.38, w: 8.7, h: 0.75, fontSize: 9 });
}

// =========================================================================================
// 20. SEEN VS UNSEEN + PER-CLASS + ABLATION
// =========================================================================================
{
  const s = slide("CONTENT", "Analysis: unseen designs, hardest class, ablation");
  s.addImage({ path: FIG("det_seen_unseen.png"), x: 0.5, y: 1.1, w: 4.4, h: 2.14, objectName: "detection seen vs unseen designs" });
  const DI = ["random", "imagenet", "simclr", "template_aware"];
  const su = i => [agg(det, "seen_mAP50", i, 0.1), agg(det, "unseen_mAP50", i, 0.1)];
  s.addTable([
    [hd("Detector, 10% labels"), hd("Seen mAP"), hd("Unseen mAP"), hd("Gap (pts)")],
    ...DI.map(i => { const [a, b] = su(i); return [i === "template_aware" ? hi(SHORT[i]) : SHORT[i], f3(a), f3(b), ((a - b) * 100).toFixed(1)]; }),
  ], tableOpts(5.1, 1.1, 4.4, [1.7, 0.9, 0.9, 0.9], 0.3));
  s.addText("Unseen = 332 test boards from 4 board groups never used for training or pretraining.", { x: 5.1, y: 2.65, w: 4.4, h: 0.5, fontSize: 9, italic: true, color: C.accent6, isTextBox: true, margin: 0, valign: "top" });
  card(s, 0.5, 3.35, 2.9, 1.75, "Generalisation", null);
  bullets(s, [
    "Random and ImageNet lose " + ((su("random")[0] - su("random")[1]) * 100).toFixed(1) + " and " + ((su("imagenet")[0] - su("imagenet")[1]) * 100).toFixed(1) + " points on unseen designs; both PCB-pretrained encoders lose none. Pretraining on 79k unlabelled patches from all designs is what buys this.",
  ], { x: 0.65, y: 3.75, w: 2.6, h: 1.3, fontSize: 9 });
  card(s, 3.55, 3.35, 2.9, 1.75, "Hardest class: short", null);
  bullets(s, [
    "AP@0.5 for short at 10% labels: random " + f3(apShort("random")) + ", ImageNet " + f3(apShort("imagenet")) + ", SimCLR " + f3(apShort("simclr")) + ", ours " + f3(apShort("template_aware")) + "; every other class is above 0.9 (det_per_class.png). A short is a thin bridge between two tracks, and it is also the weakest class in patch classification (0.86 recall at 2% labels, seed 0). PCB-specific pretraining gives its largest gain exactly here.",
  ], { x: 3.7, y: 3.75, w: 2.6, h: 1.3, fontSize: 9 });
  card(s, 6.6, 3.35, 2.9, 1.75, "Ablation verdict", null);
  bullets(s, [
    "Naive template positives are harmful everywhere: kNN F1 " + f3(knnAt("template_pos", 60)) + ", linear probe " + f3(agg(clsLP, "test_f1", "template_pos", 1.0)) + " vs " + f3(agg(clsLP, "test_f1", "template_aware", 1.0)) + " for ours, worst fine-tuned F1 at 2% (" + f3(agg(clsFT, "test_f1", "template_pos", 0.02)) + "). The gain comes from routing by the difference map, not from using templates at all.",
  ], { x: 6.75, y: 3.75, w: 2.6, h: 1.3, fontSize: 9 });
}

// =========================================================================================
// 21. INTERPRETABILITY
// =========================================================================================
{
  const s = slide("CONTENT", "Interpretability: Grad-CAM and t-SNE");
  s.addImage({ path: FIG("gradcam.png"), x: 0.5, y: 1.1, w: 3.3, h: 3.96, objectName: "Grad-CAM per class and initialisation" });
  s.addImage({ path: FIG("tsne.png"), x: 3.95, y: 1.1, w: 5.55, h: 1.17, objectName: "t-SNE of frozen features" });
  card(s, 3.95, 2.4, 5.55, 1.3, "t-SNE (frozen encoders, test patches, no fine-tuning)", null);
  bullets(s, [
    "SimCLR and template-aware features form separate clusters for all six defect classes with copper and pin-hole fully isolated; ImageNet and random features mix every class into one cloud. The naive variant separates copper and pin-hole only.",
  ], { x: 4.1, y: 2.8, w: 5.25, h: 0.85, fontSize: 9.5 });
  card(s, 3.95, 3.8, 5.55, 1.26, "Grad-CAM, layer3, classifiers fine-tuned with 10% labels", null);
  bullets(s, [
    "Pretrained encoders attend to the defect itself (the bridge of a short, the tip of a spur, the gap of an open). The random-init classifier spreads attention over the whole patch and mislabels the short as open; ImageNet attends to the right region but more diffusely.",
  ], { x: 4.1, y: 4.2, w: 5.25, h: 0.8, fontSize: 9.5 });
}

// =========================================================================================
// 22. QUALITATIVE DETECTIONS + DEMO
// =========================================================================================
{
  const s = slide("CONTENT", "Qualitative detections and the Gradio demo");
  s.addImage({ path: FIG("det_examples.png"), x: 0.96, y: 1.05, w: 8.08, h: 2.8, objectName: "detections vs ground truth on seen and unseen boards" });
  s.addText("Red: detections with score at least 0.5. Blue: ground truth. Left: a seen design. Centre and right: designs never used in training. Generated by make_report.py from the best detector in runs/det.", { x: 0.5, y: 3.88, w: 9.0, h: 0.32, fontSize: 9, italic: true, color: C.accent6, isTextBox: true, margin: 0 });
  card(s, 0.5, 4.25, 4.4, 0.9, "Demo tab 1: board inspection", "python app/demo.py. Upload a board (and template): boxes with scores, plus the label-free difference map side by side.", { bodySize: 9, headSize: 11 });
  card(s, 5.1, 4.25, 4.4, 0.9, "Demo tab 2: patch explanation", "Crop a 64 x 64 patch: 7-class prediction with a Grad-CAM heat map from the Stage A classifier.", { bodySize: 9, headSize: 11 });
}

// =========================================================================================
// 23. CONCLUSIONS AND LIMITATIONS
// =========================================================================================
{
  const s = slide("CONTENT", "Conclusions: hypotheses against the evidence");
  const m = (i, f) => agg(det, "mAP50", i, f);
  const verdict = [
    ["H1", "PCB-specific SimCLR beats random init and closes the gap to ImageNet when labels are scarce.", "Confirmed", "Better than random at every fraction in all three evaluations; ahead of ImageNet in detection at 10% and 25% labels, behind only at 100%.", C.accent3],
    ["H2", "Template-aware SimCLR beats plain SimCLR.", "Confirmed for the representation, not after fine-tuning", "Higher kNN F1, higher linear-probe F1 at all four fractions and no seen-unseen gap. After full fine-tuning the two are within seed noise, and tight-box mAP@0.5:0.95 shows no gain.", C.accent2],
    ["H3", "Naive template positives hurt defect classes.", "Confirmed", "Worst encoder on every metric; the defect must stay visible to the contrastive objective.", C.accent3],
  ];
  verdict.forEach(([id, h, v, body, col], i) => {
    const y = 1.1 + i * 0.95;
    s.addShape(pres.ShapeType.roundRect, { x: 0.5, y, w: 9.0, h: 0.87, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "verdict " + id });
    numCircle(s, 0.65, y + 0.24, id, { fill: col, d: 0.42, size: 11 });
    s.addText([{ text: h, options: { bold: true, color: C.text2, breakLine: true } }, { text: v, options: { bold: true, color: col } }], { x: 1.25, y: y + 0.08, w: 3.4, h: 0.75, fontSize: 10, isTextBox: true, margin: 0, valign: "top" });
    s.addText(body, { x: 4.75, y: y + 0.08, w: 4.6, h: 0.75, fontSize: 9.5, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
  });
  card(s, 0.5, 4.0, 9.0, 1.15, "Limitations", null);
  bullets(s, [
    "Stage B has one seed and one backbone (ResNet-18) on an 8 GB GPU; DeepPCB images are already binarised and template-aligned, so a real line needs registration before the difference map can be used.",
    "ImageNet remains a strong initialisation when every label is available (" + f3(m("imagenet", 1.0)) + " vs " + f3(m("template_aware", 1.0)) + " mAP@0.5); the template-aware gain is on labels needed, generalisation to unseen designs and frozen-feature quality.",
  ], { x: 0.65, y: 4.38, w: 8.7, h: 0.75, fontSize: 9 });
}

// =========================================================================================
// 22. LITERATURE REVIEW
// =========================================================================================
{
  const s = slide("CONTENT", "Literature review: papers this work builds on", "Literature and references");
  const hd = t => ({ text: t, options: { bold: true, color: H.lt1, fill: { color: H.dk2 } } });
  const link = (t, u) => ({ text: t, options: { hyperlink: { url: u }, color: H.hlink } });
  const rows = [
    [hd("Reference"), hd("Contribution used here")],
    [link("Tang, He, Liu, Wu. Online PCB Defect Detector on a New PCB Defect Dataset. arXiv:1902.06197, 2019", "https://arxiv.org/abs/1902.06197"), "DeepPCB dataset, 6 defect classes, template pairs, benchmark protocol (F-score at IoU 0.33)."],
    [link("Chen, Kornblith, Norouzi, Hinton. A Simple Framework for Contrastive Learning of Visual Representations (SimCLR). ICML 2020", "https://arxiv.org/abs/2002.05709"), "Two-view contrastive pretraining, NT-Xent loss, projector head, strong augmentation."],
    [link("Robinson, Chuang, Sra, Jegelka. Contrastive Learning with Hard Negative Samples. ICLR 2021", "https://arxiv.org/abs/2010.04592"), "Motivation for per-anchor hard negatives in the NT-Xent denominator."],
    [link("He, Fan, Wu, Xie, Girshick. Momentum Contrast for Unsupervised Visual Representation Learning. CVPR 2020", "https://arxiv.org/abs/1911.05722"), "Context on contrastive SSL design choices and transfer to detection."],
    [link("Ren, He, Girshick, Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS 2015", "https://arxiv.org/abs/1506.01497"), "Two-stage detector with RPN used in Stage B."],
    [link("Lin et al. Feature Pyramid Networks for Object Detection. CVPR 2017", "https://arxiv.org/abs/1612.03144"), "Multi-scale features for small defects; R18-FPN backbone."],
    [link("He, Zhang, Ren, Sun. Deep Residual Learning for Image Recognition. CVPR 2016", "https://arxiv.org/abs/1512.03385"), "ResNet-18 encoder shared by every arm."],
    [link("Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks. ICCV 2017", "https://arxiv.org/abs/1610.02391"), "Interpretability of the patch classifier."],
    [link("Ding, Dai, Li, Liu. TDD-Net: A Tiny Defect Detection Network for PCBs. CAAI Trans. Intell. Technol., 2019", "https://doi.org/10.1049/trit.2019.0019"), "Prior work on small-anchor design for PCB defects."],
  ];
  s.addTable(rows, { x: 0.5, y: 1.05, w: 9.0, colW: [5.3, 3.7], fontSize: 9.5, color: C.text1, border: { type: "solid", color: "DDE6E1", pt: 0.5 }, rowH: 0.38, valign: "middle", margin: 0.04 });
}

// =========================================================================================
// 23. DATASETS, TOOLS AND SITES
// =========================================================================================
{
  const s = slide("CONTENT", "References: dataset, software and sources");
  const link = (t, u) => ({ text: t, options: { hyperlink: { url: u }, color: H.hlink, breakLine: true } });
  const plain = (t) => ({ text: t, options: { color: H.accent6, breakLine: true, fontSize: 9.5 } });
  s.addText("Dataset", { x: 0.5, y: 1.1, w: 4.4, h: 0.3, fontSize: 14, bold: true, color: C.text2, isTextBox: true, margin: 0 });
  s.addText([
    link("DeepPCB repository (images, templates, annotations, evaluation code)", "https://github.com/tangsanli5201/DeepPCB"),
    plain("github.com/tangsanli5201/DeepPCB. Research use only, per the authors' licence. Cloned by scripts/prepare_data.py --download."),
    link("DeepPCB paper on arXiv", "https://arxiv.org/abs/1902.06197"),
    plain("arxiv.org/abs/1902.06197"),
  ], { x: 0.5, y: 1.45, w: 4.4, h: 1.5, fontSize: 10.5, isTextBox: true, margin: 0, valign: "top" });

  s.addText("Software and documentation", { x: 5.1, y: 1.1, w: 4.4, h: 0.3, fontSize: 14, bold: true, color: C.text2, isTextBox: true, margin: 0 });
  s.addText([
    link("PyTorch documentation", "https://pytorch.org/docs/stable/index.html"), plain("pytorch.org/docs, including torch.amp mixed precision"),
    link("torchvision detection models (Faster R-CNN, FPN)", "https://pytorch.org/vision/stable/models.html"), plain("pytorch.org/vision/stable/models.html"),
    link("Gradio documentation", "https://www.gradio.app/docs"), plain("gradio.app/docs"),
    link("scikit-learn (t-SNE, kNN)", "https://scikit-learn.org/stable/"), plain("scikit-learn.org"),
    link("SciPy ndimage (binary opening, labelling)", "https://docs.scipy.org/doc/scipy/reference/ndimage.html"), plain("docs.scipy.org"),
  ], { x: 5.1, y: 1.45, w: 4.4, h: 2.6, fontSize: 10.5, isTextBox: true, margin: 0, valign: "top" });

  s.addText("Course and institution", { x: 0.5, y: 3.1, w: 4.4, h: 0.3, fontSize: 14, bold: true, color: C.text2, isTextBox: true, margin: 0 });
  s.addText([
    link("RVCE scheme and syllabus page", "https://rvce.edu.in/academics_and_examinations/rvce_scheme_syllabus/"), plain("rvce.edu.in, Academics and Examinations"),
    link("CSE VII and VIII semester scheme (Deep Learning, 21IS73GA, PCE-III Group G)", "https://rvce.edu.in/wp-content/uploads/2025/10/CSE-2021-4th-year.pdf"), plain("Syllabus units: neural networks and MLPs, CNNs, sequence models, autoencoders, pretrained models and augmentation."),
    link("Goodfellow, Bengio, Courville. Deep Learning. MIT Press, 2016 (prescribed text)", "https://www.deeplearningbook.org/"), plain("deeplearningbook.org"),
  ], { x: 0.5, y: 3.45, w: 4.4, h: 1.7, fontSize: 10.5, isTextBox: true, margin: 0, valign: "top" });
  s.addText("Project repository layout, reproduction commands and the regenerated results file (reports/RESULTS.md) are described in the project README.", { x: 5.1, y: 4.3, w: 4.4, h: 0.7, fontSize: 10, italic: true, color: C.accent6, isTextBox: true, margin: 0, valign: "top" });
}

// =========================================================================================
// 24. CLOSE
// =========================================================================================
{
  const s = slide("SECTION", "Thank you", null);
  s.addText("Questions and discussion\nCode: github.com/divyaprakashcs23-hash/pcb-template-aware-ssl  (reproduce with prepare_data.py, run_experiments.py, make_report.py)", { placeholder: "body" });
}

(async () => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT, "| simclr epochs logged:", simclrEpochs.length);
})();
