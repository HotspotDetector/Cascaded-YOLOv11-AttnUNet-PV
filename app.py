"""
FADNet Gradio GUI
=================
Thermal Hotspot & Crack Detection — Interactive Inference Dashboard
Supports: Standard, Multi-Resolution WBF, and SAHI inference modes.

Run:
    pip install gradio ultralytics ensemble-boxes opencv-python-headless
    python app.py
"""

import os, sys, math, cv2, pathlib, warnings, textwrap
import numpy as np
import gradio as gr
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Constants & Paths  (edit these to match your environment)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
CKPT_DIR   = BASE_DIR / "checkpoints"

CHECKPOINTS = {
    "FADNet Finetune (Best)":  str(CKPT_DIR / "fadnet_finetune_best.pt"),
    "FADNet YOLO Backbone":    str(CKPT_DIR / "fadnet_yolo_best.pt"),
}

CLASS_NAMES  = ["Hotspot", "Crack"]
N_CLASSES    = 2

# F1-optimal defaults (from notebook Cell 19/20)
DEFAULT_CONF_HOTSPOT = 0.20
DEFAULT_CONF_CRACK   = 0.20

# Colour palette  (BGR → used by cv2, converted to RGB for Gradio)
COLORS = {
    "Hotspot": (255,  80,  60),   # bright red-orange
    "Crack":   ( 60, 140, 255),   # cornflower blue
    "GT":      (  0, 220,   0),   # green
    "TP":      (  0, 200, 200),   # cyan
    "FP":      (  0,   0, 220),   # red
    "FN":      (  0, 200, 220),   # yellow-ish
}

GALLERY_IMAGES = sorted((BASE_DIR / "working").glob("*.png")) if (BASE_DIR / "working").exists() else []

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CoordAtt Patch  (required before loading any FADNet checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

class h_sigmoid(nn.Module):
    def forward(self, x): return nn.functional.relu6(x + 3) / 6

class h_swish(nn.Module):
    def forward(self, x): return x * h_sigmoid()(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup=None, reduction=32):
        super().__init__()
        oup = oup or inp
        mip = max(8, inp // reduction)
        self.conv1  = nn.Conv2d(inp, mip, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(mip)
        self.act    = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        xh = x.mean(dim=3, keepdim=True)
        xw = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y  = torch.cat([xh, xw], dim=2)
        y  = self.act(self.bn1(self.conv1(y)))
        xh, xw = torch.split(y, [H, W], dim=2)
        xw = xw.permute(0, 1, 3, 2)
        return x * torch.sigmoid(self.conv_h(xh)) * torch.sigmoid(self.conv_w(xw))


def patch_ultralytics():
    """Inject CoordAtt into Ultralytics in-memory only — no files written to disk."""
    try:
        import ultralytics.nn.modules as M
        import ultralytics.nn.tasks as T

        # Register CoordAtt on the live module objects (in-memory only)
        M.CoordAtt = CoordAtt
        T.CoordAtt = CoordAtt

        # Create a fake sub-module so pickle/torch.load can resolve the class path
        fake_mod = type(sys)("ultralytics.nn.modules.coord_att")
        fake_mod.CoordAtt  = CoordAtt
        fake_mod.h_swish   = h_swish
        fake_mod.h_sigmoid = h_sigmoid
        sys.modules["ultralytics.nn.modules.coord_att"] = fake_mod

        # Also register under the tasks module's namespace
        T.__dict__["CoordAtt"] = CoordAtt

        return True, "CoordAtt patch applied (in-memory) ✓"
    except Exception as e:
        return False, f"Patch failed: {e}"
# Apply patch at startup
_patch_ok, _patch_msg = patch_ultralytics()
print(_patch_msg)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Model Cache
# ─────────────────────────────────────────────────────────────────────────────
_model_cache: dict[str, object] = {}

def load_model(ckpt_name: str):
    """Load (and cache) a YOLO checkpoint by friendly name."""
    from ultralytics import YOLO

    ckpt_path = CHECKPOINTS.get(ckpt_name)
    if not ckpt_path:
        raise ValueError(f"Unknown checkpoint: {ckpt_name}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at:\n  {ckpt_path}\n\n"
            "Copy the .pt files into the checkpoints/ folder next to app.py."
        )
    if ckpt_name not in _model_cache:
        _model_cache[ckpt_name] = YOLO(ckpt_path)
    return _model_cache[ckpt_name]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_box(img, x1, y1, x2, y2, color_bgr, label, font_scale=0.48, thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    by = max(y1 - 4, th + 4)
    cv2.rectangle(img, (x1, by - th - 4), (x1 + tw + 6, by), color_bgr, -1)
    cv2.putText(img, label, (x1 + 3, by - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_image(img_bgr, boxes_norm, scores, labels,
                   conf_thrs=(0.20, 0.20), draw_conf=True):
    """
    Draw predicted bounding boxes on a BGR image copy.
    Returns an RGB numpy array.
    boxes_norm : list of [x1,y1,x2,y2] in [0,1]
    """
    vis = img_bgr.copy()
    H, W = vis.shape[:2]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    for i in order:
        lbl   = labels[i]
        score = scores[i]
        if score < conf_thrs[lbl]:
            continue
        box = boxes_norm[i]
        x1, y1 = int(box[0] * W), int(box[1] * H)
        x2, y2 = int(box[2] * W), int(box[3] * H)
        col   = COLORS[CLASS_NAMES[lbl]]
        text  = f"{CLASS_NAMES[lbl]} {score:.2f}" if draw_conf else CLASS_NAMES[lbl]
        _draw_box(vis, x1, y1, x2, y2, col, text)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Inference Modes
# ─────────────────────────────────────────────────────────────────────────────

def _yolo_predict(model, img_path_or_arr, imgsz, conf_raw, iou_raw, device):
    """Run YOLO.predict and return (boxes_norm, scores, labels)."""
    is_arr = isinstance(img_path_or_arr, np.ndarray)
    src    = img_path_or_arr

    # Get image dims for normalisation
    if is_arr:
        H, W = src.shape[:2]
    else:
        tmp = cv2.imread(str(img_path_or_arr))
        H, W = tmp.shape[:2]

    res = model.predict(
        src, imgsz=imgsz, conf=conf_raw, iou=iou_raw,
        verbose=False, save=False, device=device,
    )
    r = res[0]
    boxes, scores, labels = [], [], []
    if len(r.boxes):
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            boxes.append([
                max(0.0, x1 / W), max(0.0, y1 / H),
                min(1.0, x2 / W), min(1.0, y2 / H),
            ])
            scores.append(float(box.conf[0]))
            # Label flip: model cls 0→dataset 1 and vice-versa
            labels.append(1 - int(box.cls[0]))
    return boxes, scores, labels


def infer_standard(model, img_bgr, conf_hotspot, conf_crack, nms_iou, imgsz, device):
    """Single-resolution inference."""
    boxes, scores, labels = _yolo_predict(
        model, img_bgr, imgsz, conf_raw=0.01, iou_raw=nms_iou, device=device
    )
    # Apply per-class threshold
    thrs = [conf_hotspot, conf_crack]
    keep = [(b, s, l) for b, s, l in zip(boxes, scores, labels) if s >= thrs[l]]
    if keep:
        b, s, l = zip(*keep)
        return list(b), list(s), list(l)
    return [], [], []


def infer_multires_wbf(model, img_bgr, conf_hotspot, conf_crack,
                       nms_iou, imgsz_list, wbf_iou, wbf_skip, device):
    """Multi-resolution Weighted Box Fusion (Lever 3 from notebook)."""
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        raise ImportError("Install ensemble-boxes:  pip install ensemble-boxes")

    all_boxes, all_scores, all_labels = [], [], []
    for imgsz in imgsz_list:
        b, s, l = _yolo_predict(model, img_bgr, imgsz, 0.01, 0.99, device)
        all_boxes.append(b); all_scores.append(s); all_labels.append(l)

    final_boxes, final_scores, final_labels = [], [], []
    for cls_id in range(N_CLASSES):
        cb = [[bx for bx, lb in zip(mb, ml) if lb == cls_id]
              for mb, ml in zip(all_boxes, all_labels)]
        cs = [[sc for sc, lb in zip(ms, ml) if lb == cls_id]
              for ms, ml in zip(all_scores, all_labels)]
        if all(len(b) == 0 for b in cb):
            continue
        b_f, s_f, l_f = weighted_boxes_fusion(
            cb, cs, [[cls_id] * len(s) for s in cs],
            weights=[1.0] * len(imgsz_list),
            iou_thr=wbf_iou, skip_box_thr=wbf_skip,
        )
        final_boxes.extend(b_f.tolist())
        final_scores.extend(s_f.tolist())
        final_labels.extend([int(x) for x in l_f])

    thrs = [conf_hotspot, conf_crack]
    keep = [(b, s, l) for b, s, l in zip(final_boxes, final_scores, final_labels) if s >= thrs[l]]
    if keep:
        b, s, l = zip(*keep)
        return list(b), list(s), list(l)
    return [], [], []


def _generate_tiles(H, W, tile_size, overlap_ratio):
    stride = int(tile_size * (1 - overlap_ratio))
    tiles  = []
    y = 0
    while y < H:
        x = 0
        while x < W:
            x2 = min(x + tile_size, W); y2 = min(y + tile_size, H)
            x1 = max(0, x2 - tile_size); y1 = max(0, y2 - tile_size)
            tiles.append((x1, y1, x2, y2))
            if x2 == W: break
            x += stride
        if y2 == H: break
        y += stride
    return tiles


def infer_sahi(model, img_bgr, conf_hotspot, conf_crack,
               tile_size, overlap, model_imgsz, wbf_iou, wbf_skip,
               full_weight, tile_weight, device):
    """SAHI Sliced Inference (Lever 4 from notebook)."""
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        raise ImportError("Install ensemble-boxes:  pip install ensemble-boxes")

    H, W = img_bgr.shape[:2]
    tiles = _generate_tiles(H, W, tile_size, overlap)

    all_boxes, all_scores, all_labels, all_weights = [], [], [], []

    # Full image
    fb, fs, fl = _yolo_predict(model, img_bgr, model_imgsz, 0.01, 0.99, device)
    all_boxes.append(fb); all_scores.append(fs); all_labels.append(fl)
    all_weights.append(full_weight)

    # Tiles
    for (tx1, ty1, tx2, ty2) in tiles:
        tile = img_bgr[ty1:ty2, tx1:tx2]
        tH, tW = tile.shape[:2]
        if tH < 8 or tW < 8:
            continue
        tb, ts, tl = _yolo_predict(model, tile, model_imgsz, 0.01, 0.99, device)
        # remap tile-relative coords → full image normalised
        mapped_boxes = []
        for bx in tb:
            ax1 = (bx[0] * tW + tx1) / W; ay1 = (bx[1] * tH + ty1) / H
            ax2 = (bx[2] * tW + tx1) / W; ay2 = (bx[3] * tH + ty1) / H
            mapped_boxes.append([
                max(0.0, ax1), max(0.0, ay1),
                min(1.0, ax2), min(1.0, ay2),
            ])
        all_boxes.append(mapped_boxes); all_scores.append(ts); all_labels.append(tl)
        all_weights.append(tile_weight)

    # WBF fusion
    final_boxes, final_scores, final_labels = [], [], []
    for cls_id in range(N_CLASSES):
        cb = [[bx for bx, lb in zip(mb, ml) if lb == cls_id]
              for mb, ml in zip(all_boxes, all_labels)]
        cs = [[sc for sc, lb in zip(ms, ml) if lb == cls_id]
              for ms, ml in zip(all_scores, all_labels)]
        if all(len(b) == 0 for b in cb):
            continue
        b_f, s_f, l_f = weighted_boxes_fusion(
            cb, cs, [[cls_id] * len(s) for s in cs],
            weights=all_weights,
            iou_thr=wbf_iou, skip_box_thr=wbf_skip,
        )
        final_boxes.extend(b_f.tolist()); final_scores.extend(s_f.tolist())
        final_labels.extend([int(x) for x in l_f])

    thrs = [conf_hotspot, conf_crack]
    keep = [(b, s, l) for b, s, l in zip(final_boxes, final_scores, final_labels) if s >= thrs[l]]
    if keep:
        b, s, l = zip(*keep)
        return list(b), list(s), list(l)
    return [], [], []


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main inference callback  (called by Gradio)
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    image_np,
    ckpt_name,
    infer_mode,
    conf_hotspot,
    conf_crack,
    nms_iou,
    imgsz,
    # Multi-res options
    use_736,
    wbf_iou,
    wbf_skip,
    # SAHI options
    sahi_tile,
    sahi_overlap,
    sahi_full_weight,
):
    if image_np is None:
        return None, "⚠️  Please upload an image first.", []

    # ── Resolve device ──────────────────────────────────────────────────────
    device = 0 if torch.cuda.is_available() else "cpu"

    # ── Load model ──────────────────────────────────────────────────────────
    try:
        model = load_model(ckpt_name)
    except (FileNotFoundError, ValueError) as e:
        return None, f"❌  {e}", []

    # ── Convert image ────────────────────────────────────────────────────────
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    try:
        if infer_mode == "Standard":
            boxes, scores, labels = infer_standard(
                model, img_bgr, conf_hotspot, conf_crack, nms_iou, int(imgsz), device
            )
        elif infer_mode == "Multi-Res WBF":
            res_list = [640, 736] if use_736 else [640]
            boxes, scores, labels = infer_multires_wbf(
                model, img_bgr, conf_hotspot, conf_crack,
                nms_iou, res_list, wbf_iou, wbf_skip, device
            )
        elif infer_mode == "SAHI":
            boxes, scores, labels = infer_sahi(
                model, img_bgr, conf_hotspot, conf_crack,
                int(sahi_tile), sahi_overlap, int(imgsz),
                wbf_iou, wbf_skip, sahi_full_weight, 1.0, device
            )
        else:
            return None, "Unknown inference mode.", []
    except Exception as e:
        import traceback
        return None, f"❌ Inference error:\n{traceback.format_exc()}", []

    # ── Annotate ─────────────────────────────────────────────────────────────
    thrs = [conf_hotspot, conf_crack]
    vis  = annotate_image(img_bgr, boxes, scores, labels, conf_thrs=thrs)

    # ── Build detection table ─────────────────────────────────────────────────
    rows = []
    for b, s, l in sorted(
        zip(boxes, scores, labels), key=lambda x: -x[1]
    ):
        if s < thrs[l]:
            continue
        rows.append([
            CLASS_NAMES[l],
            f"{s:.3f}",
            f"[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]",
        ])

    # ── Summary text ──────────────────────────────────────────────────────────
    n_hotspot = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= thrs[l])
    n_crack   = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= thrs[l])
    device_str = f"GPU (cuda:{device})" if device != "cpu" else "CPU"
    summary = (
        f"✅  **{n_hotspot + n_crack} detection(s)** — "
        f"{n_hotspot} Hotspot · {n_crack} Crack\n\n"
        f"Mode: `{infer_mode}` · Checkpoint: `{ckpt_name}` · Device: `{device_str}`"
    )

    return vis, summary, rows


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#0f1117",
    body_background_fill_dark="#0f1117",
    block_background_fill="#1a1e2e",
    block_background_fill_dark="#1a1e2e",
    block_border_color="#2d3148",
    block_border_color_dark="#2d3148",
    block_label_text_color="#c9d1e0",
    block_label_text_color_dark="#c9d1e0",
    input_background_fill="#22273a",
    input_background_fill_dark="#22273a",
    slider_color="#f97316",
    slider_color_dark="#f97316",
    button_primary_background_fill="#f97316",
    button_primary_background_fill_hover="#ea6a0b",
    button_primary_text_color="#ffffff",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
)

CSS = """
#title-banner {
    background: linear-gradient(135deg, #1e2235 0%, #252b42 50%, #1a1e2e 100%);
    border: 1px solid #f97316;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 8px;
}
#title-banner h1 { color: #f97316 !important; margin: 0 0 4px 0; font-size: 2rem; }
#title-banner p  { color: #94a3b8 !important; margin: 0; }

.detect-table thead th { background: #252b42 !important; color: #f97316 !important; }
.detect-table tbody tr:nth-child(even) { background: #1f2333 !important; }

.mode-card { border-left: 3px solid #f97316; padding-left: 10px; }

footer { display: none !important; }
"""

def build_ui():
    with gr.Blocks(theme=THEME, css=CSS, title="FADNet — Thermal Defect Detector") as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="title-banner">
          <h1>🔥 FADNet — Thermal Defect Detector</h1>
          <p>Hotspot &amp; Crack detection in thermal images · YOLOv8 + CoordAtt ·
             mAP@0.5 = 91.51% (Multi-Res WBF)</p>
        </div>
        """)

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════════
            # TAB 1 — Inference
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🎯 Inference", id="infer"):
                with gr.Row(equal_height=False):

                    # ── LEFT COLUMN — Settings ─────────────────────────────
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### ⚙️ Checkpoint")
                        ckpt_radio = gr.Radio(
                            choices=list(CHECKPOINTS.keys()),
                            value=list(CHECKPOINTS.keys())[0],
                            label="Model checkpoint",
                            show_label=False,
                        )

                        gr.Markdown("### 🧠 Inference Mode")
                        mode_radio = gr.Radio(
                            choices=["Standard", "Multi-Res WBF", "SAHI"],
                            value="Standard",
                            label="Inference mode",
                            show_label=False,
                        )
                        mode_desc = gr.Markdown(
                            "<div class='mode-card'>Single-scale inference. Fast & accurate.</div>",
                            elem_classes=["mode-card"],
                        )

                        gr.Markdown("### 🔧 Per-Class Thresholds")
                        conf_hot = gr.Slider(
                            0.01, 0.99, value=DEFAULT_CONF_HOTSPOT, step=0.01,
                            label="Hotspot confidence threshold",
                        )
                        conf_crk = gr.Slider(
                            0.01, 0.99, value=DEFAULT_CONF_CRACK, step=0.01,
                            label="Crack confidence threshold",
                        )
                        nms_iou  = gr.Slider(
                            0.10, 0.90, value=0.45, step=0.05,
                            label="NMS / WBF IoU threshold",
                        )
                        imgsz    = gr.Slider(
                            320, 1280, value=640, step=32,
                            label="Model input resolution (px)",
                        )

                        # Multi-Res options
                        with gr.Group(visible=False) as multires_group:
                            gr.Markdown("#### Multi-Res WBF Options")
                            use_736  = gr.Checkbox(value=True, label="Also run at 736 px")
                            wbf_iou  = gr.Slider(0.10, 0.80, value=0.45, step=0.05, label="WBF IoU threshold")
                            wbf_skip = gr.Slider(0.001, 0.10, value=0.001, step=0.001, label="WBF skip box threshold")

                        # SAHI options
                        with gr.Group(visible=False) as sahi_group:
                            gr.Markdown("#### SAHI Options")
                            sahi_tile    = gr.Slider(192, 512, value=320, step=32,  label="Tile size (px)")
                            sahi_overlap = gr.Slider(0.10, 0.60, value=0.40, step=0.05, label="Tile overlap ratio")
                            sahi_full_w  = gr.Slider(0.5,  3.0,  value=1.5,  step=0.1,  label="Full-image weight (vs tile=1.0)")

                        run_btn  = gr.Button("▶  Run Detection", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑  Clear", variant="secondary")

                    # ── RIGHT COLUMN — I/O ────────────────────────────────
                    with gr.Column(scale=2):
                        with gr.Row():
                            input_img  = gr.Image(
                                type="numpy", label="Input Image",
                                height=400,
                            )
                            output_img = gr.Image(
                                type="numpy", label="Detection Result",
                                height=400,
                            )

                        summary_md = gr.Markdown("*Upload an image and click **Run Detection**.*")

                        detect_table = gr.Dataframe(
                            headers=["Class", "Confidence", "Box [x1, y1, x2, y2]"],
                            datatype=["str", "str", "str"],
                            label="Detections",
                            wrap=True,
                            elem_classes=["detect-table"],
                        )

            # ══════════════════════════════════════════════════════════════════
            # TAB 2 — Analytics
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("📊 Analytics"):
                gr.Markdown("### Pre-computed Metrics from Training Run")

                CHART_META = [
                    ("fadnet_metrics_dashboard.png",  "📈 Full Metrics Dashboard"),
                    ("fadnet_advanced_push.png",       "🚀 Technique Comparison"),
                    ("perclass_thresh_heatmap.png",    "🌡️ Per-Class Threshold Heatmap"),
                    ("f1_optimal_curves.png",          "📉 F1-Optimal Threshold Curves"),
                    ("fadnet_result_grid.png",         "🖼️ Result Image Grid (GT vs Pred)"),
                    ("fadnet_live_inference.png",      "🔴 Live Inference Samples"),
                    ("fadnet_bbox_quality.png",        "🔍 Bounding Box Quality Inspector"),
                ]

                working_dir = BASE_DIR / "working"
                for fname, label in CHART_META:
                    fpath = working_dir / fname
                    if fpath.exists():
                        gr.Markdown(f"#### {label}")
                        gr.Image(value=str(fpath), label=label, show_label=False)
                    else:
                        gr.Markdown(
                            f"*`{fname}` not found — run the notebook to generate it.*"
                        )

            # ══════════════════════════════════════════════════════════════════
            # TAB 3 — Model Info
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("ℹ️ Model Info"):
                gr.Markdown("""
## FADNet — Architecture & Results

### 🏗️ Architecture
FADNet is a **YOLOv8-based thermal defect detector** enhanced with **CoordAttention (CoordAtt)**
— a coordinate-aware channel attention mechanism that captures long-range spatial dependencies
in both horizontal and vertical directions simultaneously.

| Component         | Detail                                      |
|-------------------|---------------------------------------------|
| Base architecture | YOLOv8                                      |
| Attention module  | CoordAtt (Hou et al., 2021)                 |
| Classes           | Hotspot (thermal) · Crack (structural)      |
| Input resolution  | 640 × 640 px (default)                      |
| Dataset           | Thermal-H&C (Roboflow)                      |

---

### 📋 Checkpoints

| File                       | Role                         |
|----------------------------|------------------------------|
| `fadnet_finetune_best.pt`  | **Primary** — fine-tuned FADNet (**recommended**) |
| `fadnet_yolo_best.pt`      | YOLO backbone variant         |
| `fadnet_unet_best.pth`     | U-Net segmentation head       |

---

### 📈 Benchmark Results (test set)

| Technique             | mAP@0.5 | Hotspot AP | Crack AP | Δ vs Baseline |
|-----------------------|---------|------------|----------|---------------|
| Baseline WBF          | 90.92%  | —          | —        | —             |
| Per-class threshold   | 90.40%  | —          | —        | −0.52%        |
| + Soft-NMS (σ=0.3)   | 90.60%  | —          | —        | −0.32%        |
| **Multi-res WBF** 🏆  | **91.51%** | **94.15%** | **88.86%** | **+0.59%** |
| SAHI (tile=384)       | 82.92%  | —          | —        | −8.00%        |

---

### 🔬 Inference Modes

**Standard** — Single-scale YOLO inference with per-class thresholds.
Fast, minimal overhead. Use for quick evaluation.

**Multi-Res WBF** — Runs inference at 640 px and 736 px, then fuses predictions
with Weighted Box Fusion. Achieves the best mAP@0.5 (91.51%).

**SAHI** — Sliced Adaptive Inference (Akyon et al., 2022). Divides the image into
overlapping tiles, runs the model on each, then merges with WBF. Best for detecting
very small hotspots in high-resolution images.

---

### 🎛️ F1-Optimal Thresholds (paper settings)
```
crack_conf   = 0.20
hotspot_conf = 0.20
mAP@0.5      = 0.9151
mean F1      = ~0.88
```
                """)

        # ── Event Wiring ────────────────────────────────────────────────────

        MODE_DESCS = {
            "Standard":     "<div class='mode-card'>Single-scale inference at your chosen resolution. Fast &amp; accurate.</div>",
            "Multi-Res WBF":"<div class='mode-card'>Runs at 640 &amp; 736 px, fuses with WBF — <strong>best mAP@0.5 (91.51%)</strong>.</div>",
            "SAHI":         "<div class='mode-card'>Slices image into overlapping tiles. Best for small hotspots in high-res images.</div>",
        }

        def on_mode_change(mode):
            return (
                MODE_DESCS[mode],
                gr.update(visible=(mode == "Multi-Res WBF")),
                gr.update(visible=(mode == "SAHI")),
            )

        mode_radio.change(
            on_mode_change,
            inputs=mode_radio,
            outputs=[mode_desc, multires_group, sahi_group],
        )

        run_btn.click(
            run_inference,
            inputs=[
                input_img, ckpt_radio, mode_radio,
                conf_hot, conf_crk, nms_iou, imgsz,
                use_736, wbf_iou, wbf_skip,
                sahi_tile, sahi_overlap, sahi_full_w,
            ],
            outputs=[output_img, summary_md, detect_table],
        )

        clear_btn.click(
            lambda: (None, None, "*Upload an image and click **Run Detection**.*", []),
            outputs=[input_img, output_img, summary_md, detect_table],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
    )
