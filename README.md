# FADNet — Fault-Aware Detection Network for Photovoltaic Thermal Imaging
 
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-green)
![mAP@0.5](https://img.shields.io/badge/mAP%400.5-91.51%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-success)
 
> **Course:** CSE300 Mini Project (Batch M46)
> **Institution:** SASTRA Deemed to be University, Srinivasa Ramanujan Center
 
---
 
## 📌 Abstract
 
In solar power systems, Photovoltaic (PV) module hotspots dramatically lower efficiency and pose safety hazards. Traditional methods like I-V curve analysis, image-level classification, and simple bounding box detection often struggle to pinpoint the precise location and size of these faults.
 
This project implements **FADNet**, a fault-aware detection network built on **YOLOv11** with a custom **CoordAttention (CoordAtt)** backbone — a coordinate-aware channel attention mechanism that captures long-range spatial dependencies in both horizontal and vertical directions simultaneously.
 
The final system achieves **mAP@0.5 of 91.51%** on the Thermal-H&C dataset using Multi-Resolution Weighted Box Fusion inference, exceeding the HOTSPOT-YOLO paper benchmark of 90.8%.
 
---
 
## 👥 Team Members
 
| Name | Role |
|:---|:---|
| **Ashvathram B** | Developer |
| **Swaminathan S** | Developer |
| **Vishokbadri K** | Developer |
 
**Project Supervisor:** Dr. Sangeetha J
 
---
 
## 🏗️ Architecture
 
FADNet is a **YOLOv11-based thermal defect detector** enhanced with **CoordAttention (CoordAtt)**.
 
| Component | Detail |
|:---|:---|
| Base architecture | YOLOv11m |
| Attention module | CoordAtt (Hou et al., 2021) |
| Classes | Hotspot · Crack |
| Input resolution | 640 × 640 px (standard) · 736 × 736 px (high-res pass) |
| Dataset | Thermal-H&C (Roboflow) |
| Inference | Multi-Resolution Weighted Box Fusion |
 
---
 
## 📈 Benchmark Results
 
| Technique | mAP@0.5 | Hotspot AP | Crack AP | Δ vs Baseline |
|:---|:---:|:---:|:---:|:---:|
| Baseline WBF | 90.92% | — | — | — |
| Per-class threshold | 90.40% | — | — | −0.52% |
| + Soft-NMS (σ=0.3) | 90.60% | — | — | −0.32% |
| **Multi-Res WBF** 🏆 | **91.51%** | **94.15%** | **88.86%** | **+0.59%** |
| SAHI (tile=384) | 82.92% | — | — | −8.00% |
 
**F1-Optimal Thresholds:**
```
crack_conf   = 0.20
hotspot_conf = 0.20
mAP@0.5      = 0.9151
mean F1      ≈ 0.88
```
 
---
 
## 📂 Repository Structure
 
```
Cascaded-YOLOv11-AttnUNet-PV/
├── app.py                        ← Gradio inference dashboard
├── requirements.txt              ← Python dependencies
├── fadnet_training.ipynb         ← Full training notebook (Kaggle)
├── checkpoints/
│   ├── fadnet_finetune_best.pt   ← Primary model (recommended) [LFS]
│   ├── fadnet_yolo_best.pt       ← YOLO backbone variant [LFS]
│   └── fadnet_unet_best.pth      ← U-Net segmentation head [LFS]
└── working/
    ├── fadnet_metrics_dashboard.png
    ├── fadnet_advanced_push.png
    ├── fadnet_bbox_quality.png
    ├── fadnet_live_inference.png
    ├── fadnet_result_grid.png
    ├── f1_optimal_curves.png
    └── perclass_thresh_heatmap.png
```
 
---
 
## 🚀 Quick Start
 
**1. Clone the repo**
```bash
git clone https://github.com/HotspotDetector/Cascaded-YOLOv11-AttnUNet-PV.git
cd Cascaded-YOLOv11-AttnUNet-PV
```
 
**2. Install dependencies**
```bash
pip install -r requirements.txt
pip install gradio
```
 
**3. Launch the inference dashboard**
```bash
python app.py
```
 
Open `http://localhost:7860` in your browser. Upload a thermal PV image and click **Run Detection**.
 
> **Note:** Checkpoints are stored via Git LFS. Make sure `git lfs` is installed before cloning, or download the `.pt`/`.pth` files manually and place them in `checkpoints/`.
 
---
 
## 🔬 Inference Modes
 
**Standard** — Single-scale YOLO inference with per-class thresholds. Fast, minimal overhead.
 
**Multi-Res WBF** — Runs inference at 640 px and 736 px, then fuses predictions with Weighted Box Fusion. Achieves best mAP@0.5 (91.51%).
 
**SAHI** — Sliced Adaptive Inference (Akyon et al., 2022). Divides the image into overlapping tiles, runs the model on each, then merges with WBF. Best for very small hotspots in high-resolution images; not recommended for standard 640 px thermal inputs.
 
---
 
## 📊 Results
 
### Metrics Dashboard
![Metrics Dashboard](working/fadnet_metrics_dashboard.png)
 
### Technique Comparison
![Advanced Push](working/fadnet_advanced_push.png)
 
### F1-Optimal Threshold Curves
![F1 Curves](working/f1_optimal_curves.png)
 
### Per-Class Threshold Heatmap
![Threshold Heatmap](working/perclass_thresh_heatmap.png)
 
### Result Grid (GT vs Pred)
![Result Grid](working/fadnet_result_grid.png)
 
### Live Inference Samples
![Live Inference](working/fadnet_live_inference.png)
 
### Bounding Box Quality
![BBox Quality](working/fadnet_bbox_quality.png)
 
---
 
## 🛠️ Methodology
 
### 1. Image Pre-processing
- **Gaussian Blurring** — reduces thermal noise
- **Image Sharpening** — improves edge definition and temperature contrast
### 2. CoordAttention Backbone
 
CoordAtt decomposes global average pooling into two 1D pooling operations along H and W axes, producing channel-aware attention maps that encode precise positional information — critical for localizing small hotspot regions in thermal imagery.
 
The decomposition works as follows:
 
```
Input Feature (C × H × W)
    │
    ├─→ Horizontal Pool: (C × H × 1)   [encodes vertical position]
    │
    └─→ Vertical Pool:   (C × 1 × W)   [encodes horizontal position]
         │
         Concat → Conv → BN → ReLU → Split
         │
         ├─→ Conv_h → Sigmoid → h_attention
         └─→ Conv_w → Sigmoid → w_attention
              │
              Elementwise multiply with input
```
 
This preserves directional positional information across both axes — unlike CBAM, which loses precise spatial context during its global pooling step.
 
### 3. Multi-Resolution WBF Inference
 
Runs the model at 640 px and 736 px simultaneously, then fuses all predictions using Weighted Box Fusion with per-class confidence thresholds. This recovers detections that single-scale inference misses due to stride aliasing — defects that fall in "dead zones" between anchor cells at one resolution are reliably captured at the other.
 
### 4. Cascade Segmentation (FAUNet)
 
Detected bounding boxes from YOLO are cropped from the original image and passed to **FAUNet1cls** — a single-class Attention U-Net — for pixel-level hotspot segmentation. This enables precise measurement of fault size, shape, and severity beyond what bounding boxes alone can convey.
 
```
Input Thermal Image
        │
        ▼
   YOLOv11m + CoordAtt
   (Multi-Res WBF)
        │
   Bounding Boxes (conf ≥ 0.20)
        │
        ▼
   Crop RoI patches
        │
        ▼
   FAUNet1cls
   (binary segmentation per crop)
        │
   Per-crop pixel masks
        │
        ▼
   Project masks → original image coordinates
        │
        ▼
   Detection boxes + Segmentation overlays
```
 
---
 
## 🗓️ Development Log
 
> A complete chronicle of every training session, architectural decision, debugging war, and breakthrough — from a 78% Roboflow baseline to 91.51% mAP@0.5.
 
---
 
### Day 1 — Thursday, February 19, 2026 | Environment Setup & First Training Session
 
**Environment:** NVIDIA RTX A4000 (17.2 GB VRAM), CUDA, Jupyter Notebook (local)
 
#### Environment Setup
 
Installed the full dependency stack:
 
```
ultralytics, roboflow, opencv-python, torch, torchvision, matplotlib, pyyaml
```
 
GPU availability confirmed — RTX A4000 with 17.2 GB VRAM detected via `torch.cuda.is_available()`. CUDA version locked to match PyTorch build.
 
#### Dataset Acquisition
 
Connected to Roboflow workspace and attempted API-based download of the `solar-thermal-hotspot` dataset (Version 1, YOLOv11 format). The API download hung repeatedly — likely a network timeout on the workspace endpoint. Switched to the locally downloaded dataset folder directly.
 
| Split | Images |
|:---|:---:|
| Train | 1,923 |
| Validation | 205 |
| Test | ~200 |
 
`data.yaml` was shipped with relative paths, which caused Ultralytics to fail silently when launched from a different working directory. Rewrote all paths to absolute using Python's `yaml` library.
 
Classes: `Hotspot`, `Crack` (two-class thermal IR detection).
 
#### Roboflow Cloud Training (250 epochs)
 
Trained YOLOv11m directly on the Roboflow platform for a cloud baseline before local work.
 
| Metric | Value |
|:---|:---|
| mAP@50 (Validation) | 78.0% |
| mAP@50 (Test) | 86.0% |
| Box Loss | ~1.55 (converged) |
| Class Loss | ~1.0 (converged) |
| DFL Loss | ~1.15 (converged) |
 
All three losses converged smoothly with no overfitting across 250 epochs. The 8-point gap between val and test mAP indicated the model was generalizing well to the test distribution — likely a distribution quirk in the Roboflow partition.
 
#### Local Training — YOLOv11m Fine-Tuning
 
Fine-tuned `yolo11m.pt` pretrained weights on the thermal PV dataset:
 
```python
model.train(
    data      = "/absolute/path/to/data.yaml",
    epochs    = 100,
    imgsz     = 640,
    batch     = 16,
    optimizer = 'Adam',
    lr0       = 0.001,
    cos_lr    = True,
    device    = 0,
    workers   = 4,
    patience  = 20
)
```
 
**Observations:**
- GPU memory: ~9.3 GB / 17.2 GB during forward pass
- Speed: ~2.7 iterations/second · ~50 seconds/epoch
- mAP in first 2 epochs: **30.9% → 42.3%** — rapid early convergence from pretrained weights
#### Issues Faced & Fixes
 
| Issue | Fix |
|:---|:---|
| Roboflow API download hanging | Used locally downloaded dataset directly |
| `data.yaml` relative path errors | Rewrote all paths to absolute via Python `yaml` |
| Training frozen/stuck in Jupyter | Cleared `.cache` files + restarted kernel |
| GPU memory overflow from multiple runs | `torch.cuda.empty_cache()` + kernel restart |
| `IndentationError` in cleanup cell | Fixed Python indentation in `if`-block |
| Slow training with `batch=8` | Switched to `batch=16` for ~2× speed improvement |
 
**Output:** `best.pt` saved under `hotspot_yolo/final/weights/`. This became the starting point for all subsequent stages.
 
---
 
### Phase 1 — Architecture Research & HOTSPOT-YOLO Dissection
 
Before committing to CoordAtt, we reverse-engineered the HOTSPOT-YOLO paper (Liu et al., 2024) to assess whether replicating their EfficientNet-B0 + CBAM + BiFPN + YOLOv11 head was feasible within our compute budget.
 
#### Architecture Dissection
 
The paper's title implied a fully custom detection head, but careful reading revealed that the architecture **kept the Ultralytics PANet + anchor-free head completely intact**. The only modification was swapping the backbone from YOLOv11's default CSPDarknet to EfficientNet-B0, with CBAM inserted at the feature pyramid connections.
 
**Critical insight:** The paper's BiFPN was not replacing PANet — it was an additional feature refinement layer inserted *before* PANet. This distinction is subtle and easy to misread from the architecture diagram alone.
 
#### Decision: CoordAttention over CBAM
 
| Mechanism | Approach | Limitation |
|:---|:---|:---|
| CBAM (Woo et al., 2018) | Channel attention → spatial attention, sequential | Loses precise positional information during global pooling |
| **CoordAtt (Hou et al., 2021)** | Two 1D pooling ops along H and W axes separately | **Preserves directional positional information** |
 
CoordAtt was selected because thermal hotspots are often long, thin, and directionally asymmetric. CoordAtt's ability to encode "where along each axis" a feature activates maps directly onto this geometry — channel-only attention (SE blocks) or sequential spatial attention (CBAM) cannot represent this.
 
---
 
### Phase 2 — CoordAttention Integration & Monkey-Patching Protocol
 
Integrating CoordAtt into Ultralytics required monkey-patching the Ultralytics module registry — modifying the model namespace and YAML architecture spec without forking the entire library.
 
#### The Three-Step Protocol
 
Ultralytics caches parsed modules aggressively. A naive import-and-register approach silently used the cached `.pyc` version, making it appear the patch had applied when it hadn't. The reliable protocol required three steps in strict order:
 
**Step 1 — In-memory registration:**
```python
from ultralytics.nn.modules import conv
conv.CoordAtt = CoordAtt
 
from ultralytics.nn import tasks
tasks.CoordAtt = CoordAtt
```
 
**Step 2 — On-disk YAML registration:**
Modify the Ultralytics `tasks.py` module dict to include `CoordAtt` as a recognized layer name, then update the model YAML to reference it at the appropriate C2f positions in the backbone.
 
**Step 3 — pycache invalidation:**
```bash
find /path/to/ultralytics -name "*.pyc" -delete
find /path/to/ultralytics -name "__pycache__" -type d -exec rm -rf {} +
```
 
Skipping Step 3 caused stale `.pyc` bytecode to shadow the patch. Symptom: model loads without error but CoordAtt layers are silently replaced with identity operations. The model trains normally, giving no indication the attention module is missing.
 
#### CoordAtt Placement in YOLOv11m
 
CoordAtt was inserted after the final C2f block in the backbone, before the feature pyramid neck. This placement lets the attention mechanism operate on high-level semantic features while full spatial resolution context is still accessible from earlier backbone stages.
 
---
 
### Phase 3 — Multi-Stage Training Pipeline on Kaggle
 
Local training was constrained to single-GPU memory. All Stage B and C training migrated to **Kaggle with dual T4 GPUs** (2 × 16 GB VRAM).
 
#### Kaggle-Specific Constraints (Documented)
 
| Constraint | Detail |
|:---|:---|
| `patience=0` | Required on Stage C — Kaggle sessions have hard time limits; patience-based early stopping terminates sessions before the best checkpoint is saved |
| `conf=0.25` | Inference threshold used throughout all evaluations |
| OOM guard | YOLO must be deleted and CUDA cache cleared before loading FAUNet |
| NumPy trapz shim | Newer NumPy (≥2.0) removed `np.trapz`; Ultralytics metric code uses it internally |
 
**NumPy shim:**
```python
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
```
 
**OOM guard (required before every FAUNet load):**
```python
del yolo_model
torch.cuda.empty_cache()
import gc; gc.collect()
```
 
#### Stage A — Baseline Fine-Tune
 
Fine-tuned the Day 1 `best.pt` checkpoint on the full Thermal-H&C dataset on Kaggle T4:
 
```python
model.train(
    data     = "data.yaml",
    epochs   = 100,
    imgsz    = 640,
    batch    = 32,
    lr0      = 1e-3,
    cos_lr   = True,
    patience = 20
)
```
 
**Stage A output mAP@0.5: ~87.3%** — significant jump from the local baseline, attributable to the larger batch size enabled by dual T4 VRAM.
 
#### Stage B — Pseudo-Label Enrichment
 
To expand the effective training set, ran Stage A's best checkpoint on unlabeled thermal PV images to generate pseudo-labels. Only high-confidence predictions were retained:
 
```python
results = model.predict(unlabeled_dir, conf=0.5, save=False)
for r in results:
    if len(r.boxes) > 0:
        save_pseudo_label(r)
```
 
Merged pseudo-labeled images with the original dataset and retrained from the Stage A checkpoint:
 
```python
model.train(
    data     = "data_augmented.yaml",
    epochs   = 60,
    imgsz    = 640,
    batch    = 24,
    lr0      = 5e-4,       # reduced LR for fine-tuning over merged data
    cos_lr   = True,
    patience = 15
)
```
 
**Stage B output mAP@0.5: ~89.1%** — pseudo-labels provided meaningful diversity signal, especially for the Crack class which was underrepresented in the original splits.
 
#### Stage C — High-Resolution Fine-Tune
 
Stage C targeted the final push past 90% by training at 736 px input resolution with a tightly controlled learning rate.
 
**Critical bug discovered:** A learning rate discontinuity caused by improper warm-up restart when loading the Stage B checkpoint. The optimizer state was re-initialized at `lr0` instead of continuing from the scheduler's final LR, causing the first ~15 epochs of Stage C to effectively re-warm-up from scratch — wasting ~30% of the training budget.
 
**Fix:** explicitly carry the optimizer state from Stage B and disable `warmup_epochs`:
 
```python
model.train(
    data          = "data_augmented.yaml",
    epochs        = 50,
    imgsz         = 736,
    batch         = 16,         # smaller batch for 736px at T4 VRAM limits
    lr0           = 1e-4,       # already near optimal — very low LR
    warmup_epochs = 0,          # CRITICAL: no re-warmup on checkpoint resume
    patience      = 0,          # CRITICAL: no early stopping on Kaggle sessions
    cos_lr        = True
)
```
 
**Stage C output mAP@0.5 (single-scale, conf=0.25): ~90.6%**
 
#### Stage C2 — Extended Fine-Tune
 
Initial Stage C2 ran only 20 epochs — the loss curve had not yet flattened, meaning training was cut off mid-improvement. Extended to 40 epochs with the same configuration. The Stage C2 checkpoint became the base for all WBF inference experiments.
 
---
 
### Phase 4 — FAUNet Integration (Cascade Segmentation Pipeline)
 
Alongside the YOLO detection pipeline, a parallel cascade architecture was developed: YOLO detects hotspot regions → crops are passed to FAUNet for pixel-level segmentation.
 
#### FAUNet Architecture (FAUNet1cls)
 
FAUNet is a single-class U-Net variant:
- **Encoder:** 4 downsampling blocks with double-conv + BatchNorm + ReLU
- **Bottleneck:** 512-channel double-conv
- **Decoder:** 4 upsampling blocks with skip connections (bilinear upsampling + concat from matching encoder levels)
- **Output:** 1×H×W sigmoid map (binary hotspot mask)
Trained on cropped RoI patches extracted from YOLO detections on the training set, with binary masks generated from the bounding box annotations (filled rectangle approximation — adequate for thermal imagery where hotspots are typically compact blobs).
 
#### OOM Management
 
Attempting to load YOLO and FAUNet simultaneously caused T4 OOM. Sequential load was non-negotiable:
 
```python
# Phase 1: run YOLO detection, collect boxes
detections = run_yolo(images)
 
# Phase 2: free YOLO, load FAUNet
del yolo_model
torch.cuda.empty_cache()
gc.collect()
 
unet = FAUNet1cls(in_channels=3, out_channels=1).to(device)
unet.load_state_dict(torch.load("fadnet_unet_best.pth"))
 
# Phase 3: run segmentation on cropped RoIs
masks = run_unet(detections, images)
```
 
---
 
### Phase 5 — Inference Optimization
 
#### Technique Grid Search
 
| Technique | mAP@0.5 | Notes |
|:---|:---:|:---|
| Baseline WBF (640px) | 90.92% | Single-scale, IoU threshold = 0.5 |
| Per-class threshold tuning | 90.40% | `crack=0.20`, `hotspot=0.20` — marginally hurt mAP |
| + Soft-NMS (σ=0.3) | 90.60% | Marginal recovery; not ideal for dense defect patterns |
| **Multi-Res WBF (640 + 736px)** 🏆 | **91.51%** | Fuses two scales; recovers stride-aliased detections |
| SAHI (tile=384, overlap=0.2) | 82.92% | Severe regression — overkill for 640px thermal inputs |
 
#### Multi-Resolution WBF — Implementation
 
```python
# Run inference at two scales
preds_640 = model.predict(img, imgsz=640, conf=0.25)
preds_736 = model.predict(img, imgsz=736, conf=0.25)
 
# Convert to WBF format [x1_norm, y1_norm, x2_norm, y2_norm]
boxes_all  = [normalize_boxes(preds_640), normalize_boxes(preds_736)]
scores_all = [preds_640.conf.cpu(), preds_736.conf.cpu()]
labels_all = [preds_640.cls.cpu(), preds_736.cls.cpu()]
 
from ensemble_boxes import weighted_boxes_fusion
boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
    boxes_all, scores_all, labels_all,
    iou_thr=0.5, skip_box_thr=0.25
)
```
 
**Why it works:** The 640 px and 736 px models have different effective strides. Defects that fall in a "dead zone" between anchor cells at 640 px are often captured cleanly at 736 px, and vice versa. WBF then averages box coordinates weighted by confidence, producing tighter, more consistently placed predictions than either model alone.
 
#### SAHI Post-Mortem
 
SAHI slices the image into overlapping tiles and runs inference on each tile. The assumption is that the model performs better when defects are large relative to the tile. For 640 px thermal images — where defects are already moderate-to-large relative to image size — slicing introduced more edge-artifact false positives than it recovered, costing 8 mAP points. SAHI is retained as an optional inference mode for high-resolution inputs (drone imagery, multi-panel setups).
 
---
 
### Phase 6 — F1 Threshold Optimization
 
After fixing the best inference mode, swept confidence thresholds per class to find F1-optimal operating points.
 
```python
thresholds = np.arange(0.05, 0.80, 0.05)
for crack_thr in thresholds:
    for hotspot_thr in thresholds:
        mean_f1 = evaluate(crack_thr, hotspot_thr)
        results.append((crack_thr, hotspot_thr, mean_f1))
```
 
Results visualized as a per-class threshold heatmap (see `working/perclass_thresh_heatmap.png`).
 
**Optimal thresholds:**
```
crack_conf   = 0.20
hotspot_conf = 0.20
mean F1      ≈ 0.88
```
 
Both classes converged to the same optimal threshold of 0.20, notably lower than the default 0.25. This reflects the relatively low defect prevalence per image — a more permissive threshold recovers true positives without dramatically increasing false positive rate at this precision-recall operating point.
 
---
 
### Phase 7 — Gradio Inference Dashboard
 
Built a Gradio-based inference dashboard (`app.py`) for interactive evaluation and demonstration.
 
**Features:**
- Upload thermal PV image → click **Run Detection**
- Three selectable inference modes: `Standard` | `Multi-Res WBF` | `SAHI`
- Per-detection confidence and class label overlaid on output image
- Class-wise confidence bar charts
- Side-by-side original vs annotated image view
**Kaggle deployment via ngrok:**
 
Gradio's `share=True` occasionally fails behind Kaggle's network proxy. Deployed using ngrok tunneling instead:
 
```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
public_url = ngrok.connect(7860)
print(f"Dashboard live at: {public_url}")
 
demo.launch(server_port=7860, share=False)
```
 
---
 
### Phase 8 — Final Validation & Benchmark
 
#### Full Training Stage Progression
 
| Stage | mAP@0.5 | Key Change |
|:---|:---:|:---|
| Roboflow Cloud Baseline | 86.0% | Day 1, 250 epochs cloud training |
| Stage A (Kaggle fine-tune) | ~87.3% | Larger batch on dual T4 |
| Stage B (+ Pseudo-Labels) | ~89.1% | Merged unlabeled thermal imagery |
| Stage C (736px, LR fix) | ~90.6% | High-res + LR discontinuity bug fixed |
| Baseline WBF (Stage C2) | 90.92% | Extended C2, single-scale WBF |
| **Multi-Res WBF** 🏆 | **91.51%** | 640px + 736px fused predictions |
| HOTSPOT-YOLO (paper) | 90.8% | Liu et al., 2024 benchmark |
 
#### Per-Class Analysis
 
- **Hotspot AP (94.15%):** Strong performance driven by hotspot's consistent thermal signature — bright, blob-like, high contrast against panel background. CoordAtt's directional pooling reinforces the radial heat distribution pattern.
- **Crack AP (88.86%):** Lower but solid. Cracks are thin, low-contrast, and directionally variable — intrinsically harder for anchor-free detectors. CoordAtt's horizontal/vertical encoding recovers many crack detections that standard channel attention would miss by failing to encode elongation direction.
#### Checkpoints
 
| Checkpoint | Description |
|:---|:---|
| `fadnet_finetune_best.pt` | Primary model — Stage C2 best weights (recommended) |
| `fadnet_yolo_best.pt` | Stage A YOLO variant (lighter, ~1% lower mAP) |
| `fadnet_unet_best.pth` | FAUNet1cls segmentation head weights |
 
All stored via **Git LFS** (files >100 MB each).
 
---
 
### Known Constraints & Environment Notes
 
| Constraint | Detail |
|:---|:---|
| Platform | Kaggle Dual T4 (2 × 16 GB VRAM) for Stages B–C; RTX A4000 local for Stage A |
| `patience=0` | Required for Stage C on Kaggle — prevents early termination of time-limited sessions |
| `conf=0.25` | Base inference threshold; superseded by 0.20 post F1 sweep |
| NumPy trapz shim | Required for NumPy ≥2.0: `np.trapz = np.trapezoid` |
| OOM guard | YOLO deleted + CUDA cache cleared before every FAUNet load |
| pycache invalidation | Required after every CoordAtt patch update — stale `.pyc` silently shadows the patch |
| CoordAtt monkey-patch | 3-step: in-memory registration → on-disk YAML edit → pycache clear |
 
---
 
### Timeline
 
| Date | Milestone |
|:---|:---|
| Feb 19, 2026 | Day 1: environment setup, Roboflow cloud baseline (86% test mAP), local training initiated |
| Feb–Mar 2026 | Architecture research: HOTSPOT-YOLO dissection, CoordAtt selection over CBAM |
| Mar 2026 | CoordAtt monkey-patching protocol finalized; Stage A training on Kaggle (87.3%) |
| Mar–Apr 2026 | Pseudo-label pipeline (Stage B, 89.1%); cascade FAUNet development |
| Apr 2026 | Stage C high-res training; LR discontinuity bug identified and fixed |
| Apr 23, 2026 | Stage C2 extended; Multi-Res WBF implemented and optimized |
| Apr 23, 2026 | **Final result: 91.51% mAP@0.5** — exceeds HOTSPOT-YOLO benchmark by +0.71% |
| Apr–May 2026 | Gradio dashboard, Git LFS checkpoint upload, README finalized |
 
---
 
## 📚 References
 
Hou, Q., Zhou, D., & Feng, J. (2021). Coordinate Attention for Efficient Mobile Network Design. *CVPR*.
 
Liu, B., et al. (2024). A Hot Spot Identification Approach for Photovoltaic Module Based on Enhanced U-Net With Squeeze-and-Excitation and VGG19. *IEEE Transactions on Instrumentation and Measurement*.
 
Akyon, F. C., et al. (2022). Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection. *ICIP*.
 
Ultralytics YOLOv11 (2024). https://docs.ultralytics.com/models/yolo11/
 
---
 
## 📝 License
 
MIT License — see [LICENSE](LICENSE) for details.
This project is developed for academic purposes at SASTRA Deemed to be University
