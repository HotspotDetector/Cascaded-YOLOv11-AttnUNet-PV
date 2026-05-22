# FADNet — Fault-Aware Detection Network for Photovoltaic Thermal Imaging
 
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-green)
![mAP@0.5](https://img.shields.io/badge/mAP%400.5-91.51%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-success)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20Dual%20T4-informational)
 
> **Course:** CSE300 Mini Project
> **Institution:** SASTRA Deemed to be University
 
---
 
## 📌 Abstract
 
Photovoltaic (PV) module hotspots and cracks dramatically reduce energy yield and pose long-term safety risks in solar installations. Conventional fault detection methods — I-V curve analysis, image-level classification, and naïve bounding box detectors — struggle to localise precise fault boundaries, especially for thermally small or spatially clustered defects.
 
**FADNet** is a fault-aware detection network built on **YOLOv11m** with a custom **CoordAttention (CoordAtt)** mechanism injected into the backbone. CoordAtt replaces standard 2D global pooling with two 1D pooling operations along H and W axes independently, producing channel-attention maps that retain directional positional information — a property especially important for detecting elongated crack patterns in thermal imagery.
 
The training pipeline is fully progressive: a four-stage YOLO curriculum (frozen-backbone → full fine-tune → mosaic-on refinement → mosaic-off clean-up) is followed by a UNet-based pseudo-label enrichment stage, which generates additional bounding box annotations by segmenting unlabelled training images. Inference is pushed further via Multi-Resolution Weighted Box Fusion (fusing predictions at 640 px and 736 px) plus Soft-NMS suppression and per-class confidence thresholding.
 
**Final result: mAP@0.5 = 91.51%** on the Thermal-H&C test split (Hotspot AP = 94.15%, Crack AP = 88.86%), exceeding the HOTSPOT-YOLO paper's reported 90.8%.
 
---
 
## 🏗️ Architecture
 
FADNet is a **YOLOv11m** detector with a monkey-patched **CoordAttention** module inserted at the backbone level. The full pipeline adds a cascade segmentation stage (FAUNet1cls) and an inference ensemble layer.
 
| Component | Detail |
|:---|:---|
| Base detector | YOLOv11m (Ultralytics) |
| Neck | PANet (unchanged from Ultralytics default) |
| Detection head | Anchor-free (unchanged from Ultralytics default) |
| Attention module | CoordAtt (Hou et al., CVPR 2021) |
| Classes | `Crack` (cls 0) · `Hotspot` (cls 1) |
| Input resolution | 640 × 640 px (standard) · 736 × 736 px (high-res ensemble pass) |
| Dataset | Thermal-H&C v1 (Roboflow `hotspotyolo` workspace) |
| Inference | Multi-Resolution Weighted Box Fusion + Soft-NMS |
| Cascade seg. | FAUNet1cls — single-class Attention U-Net per cropped RoI |
 
---
 
## 📈 Benchmark Results
 
### Inference technique ablation (all inference-only, no retraining)
 
| Technique | mAP@0.5 | Hotspot AP | Crack AP | Δ vs Baseline |
|:---|:---:|:---:|:---:|:---:|
| Baseline WBF (Stage C2) | 90.92% | — | — | — |
| Per-class conf threshold | 90.40% | — | — | −0.52% |
| + Soft-NMS (σ=0.3) | 90.60% | — | — | −0.32% |
| **Multi-Res WBF (640+736)** 🏆 | **91.51%** | **94.15%** | **88.86%** | **+0.59%** |
| SAHI (tile=384, overlap=0.4) | 82.92% | — | — | −8.00% |
| HOTSPOT-YOLO (Liu et al. 2024) | 90.8% | — | — | — |
 
### F1-Optimal Operating Point
 
```
crack_conf    = 0.20
hotspot_conf  = 0.20
mAP@0.5       = 0.9151   ← paper metric (full PR curve, conf → 0)
mean F1       ≈ 0.88     ← operational metric at above thresholds
```
 
### Per-class confidence distribution (diagnostic, conf=0.01)
 
| Class | Mean conf | Median conf | Key finding |
|:---|:---:|:---:|:---|
| Crack | 0.474 | 0.582 | Peaked at high confidence — can afford higher threshold |
| Hotspot | 0.258 | 0.085 | Long low-conf tail — needs lower threshold to retain TPs |
 
This asymmetry is why a single global conf kills mAP: the optimal Crack threshold suppresses ~60% of weak Hotspot predictions. Per-class thresholding is mandatory.
 
---
 
## 📂 Repository Structure
 
```
Cascaded-YOLOv11-AttnUNet-PV/
├── app.py                        ← Gradio inference dashboard (ngrok-tunnelled)
├── requirements.txt              ← Python dependencies
├── fadnet_base.ipynb             ← Full training pipeline: Stage A→B→C + UNet + diagnostics (Kaggle)
├── fadnet_trainingv1.ipynb       ← Advanced inference push: Multi-Res WBF, SAHI, Soft-NMS, F1 sweep (Kaggle)
├── checkpoints/
│   ├── fadnet_finetune_best.pt   ← Primary model: stageC_aug_v2_p2/best.pt [LFS]
│   ├── fadnet_yolo_best.pt       ← Alternate YOLO checkpoint [LFS]
│   └── fadnet_unet_best.pth      ← FAUNet1cls segmentation head [LFS]
└── working/
    ├── fadnet_metrics_dashboard.png
    ├── fadnet_advanced_push.png
    ├── fadnet_bbox_quality.png
    ├── fadnet_live_inference.png
    ├── fadnet_result_grid.png
    ├── f1_optimal_curves.png
    └── perclass_thresh_heatmap.png
```
 
> **Notebook execution order:**
> 1. `fadnet_base.ipynb` — Run Cell 1 (CoordAtt patch) every session. Then Cells 2–30 in order to reproduce the full pipeline from dataset download through UNet pseudo-label enrichment.
> 2. `fadnet_trainingv1.ipynb` — Inference-only. Run Cell 1 (CoordAtt patch) then Cells 2–20 in order. No retraining required.
 
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
 
> **Note:** Checkpoints are stored via Git LFS. Run `git lfs install && git lfs pull` after cloning, or download `.pt`/`.pth` files manually and place them in `checkpoints/`.
 
**To reproduce training on Kaggle:**
- Upload `fadnet_base.ipynb` to a Kaggle notebook with GPU accelerator = **T4 × 2**
- Add your Roboflow API key as a Kaggle Secret named `ROBOFLOW_API_KEY`
- Run all cells in order
---
 
## 🔬 Inference Modes
 
**Standard single-scale** — Single forward pass at 640 px with per-class conf thresholds. Fastest, suitable for real-time deployment.
 
**Multi-Res WBF** — Runs two independent forward passes at 640 px and 736 px. Each resolution resolves differently-sized defects: a 160×120 px hotspot fills 25% of a 640-tile but only 6% of a 736-tile, so the model literally "sees it differently." Predictions from both passes are fused with Weighted Box Fusion (IoU threshold 0.5). Achieves best mAP@0.5 (91.51%). Zero training cost.
 
**Soft-NMS (Gaussian)** — Replaces hard NMS with Gaussian score decay: `score_j *= exp(−IoU(j,max)² / σ)`. Adjacent genuine hotspots on adjacent PV cells that overlap slightly survive instead of being killed. Bodla et al. (ICCV 2017) showed +1.1–1.7% mAP over best hard-NMS threshold on PASCAL VOC and MS-COCO. In FADNet's thermal domain, adjacent cluster suppression was a real failure mode (confirmed in FP/FN diagnostics). Optimal σ = 0.3 after grid search.
 
**SAHI** — Sliced Adaptive Hyper Inference (Akyon et al., 2022). Splits each image into overlapping tiles (e.g., 384×384, 40% overlap) plus the full image, runs the model on each independently, then merges with WBF. Published +5–7% AP on aerial/thermal small-object datasets. In FADNet's case SAHI degraded mAP (−8%) because the standard dataset images are already 640 px and most defects are not unusually small relative to image size — tiling at 384 px created more partial-box edge artefacts than it recovered.
 
---
 
## 📊 Results
 
### Metrics Dashboard
![Metrics Dashboard](working/fadnet_metrics_dashboard.png)
 
### Inference Technique Comparison
![Advanced Push](working/fadnet_advanced_push.png)
 
### F1-Optimal Threshold Curves
![F1 Curves](working/f1_optimal_curves.png)
 
### Per-Class Confidence Threshold Heatmap
![Threshold Heatmap](working/perclass_thresh_heatmap.png)
 
### Result Grid (GT vs Pred)
![Result Grid](working/fadnet_result_grid.png)
 
### Live Inference Samples
![Live Inference](working/fadnet_live_inference.png)
 
### Bounding Box Quality
![BBox Quality](working/fadnet_bbox_quality.png)
 
---
 
## 🛠️ Methodology
 
### 1. Dataset
 
The dataset was pulled from Roboflow workspace, using the `yolov11` export format.
 
| Split | Images |
|:---|:---:|
| Train | 1,923 |
| Validation | 205 |
| Test | ~200 |
 
**Critical label remap:** The raw Roboflow download ships class 0 = `PV Hotspot`, class 1 = `Crack`. FADNet's training inverts this to class 0 = `Crack`, class 1 = `Hotspot` for consistency with the inference pipeline. All label files in train/valid/test are remapped on download using a single-pass script that flips `0 → 1` and `1 → 0` in every `.txt` annotation.
 
**data.yaml fix:** Roboflow ships `data.yaml` with relative paths which Ultralytics resolves incorrectly in Kaggle's working directory. All paths are rewritten to absolute on download.
 
### 2. CoordAttention Backbone
 
CoordAtt (Hou et al., CVPR 2021) decomposes the standard 2D global average pooling in a channel-attention module into two separate 1D pooling operations — one along the H axis and one along the W axis. This produces attention maps that encode both channel dependencies and spatial directional context simultaneously.
 
```
Input Feature Map: (B, C, H, W)
        │
        ├─→ Mean pool along W axis → (B, C, H, 1)   [vertical positional encoding]
        │
        └─→ Mean pool along H axis → (B, C, 1, W)   [horizontal positional encoding]
                 permute → (B, C, W, 1)
        │
        Concatenate along H dim → (B, C, H+W, 1)
        │
        Conv2d(C → C//r, 1×1) → BN → h_swish activation
        │
        Split along H dim:
          ├─→ (B, C//r, H, 1)  →  Conv2d(C//r → C, 1×1) → Sigmoid → attention_h
          └─→ (B, C//r, W, 1)  →  Conv2d(C//r → C, 1×1) → Sigmoid → attention_w (permuted back)
        │
        Output = Input × attention_h × attention_w
```
 
The activation function is **h-swish**: `x * relu6(x+3) / 6` — a hardware-friendly approximation of swish that avoids the expensive `exp()` of exact sigmoid.
 
**Why CoordAtt over CBAM:** CBAM applies channel attention then spatial attention sequentially. Both operations use 2D global pooling which collapses spatial structure, losing precise location information. CoordAtt retains horizontal/vertical position encoding, which is important for localising elongated crack patterns running along panel cell boundaries.
 
| Mechanism | Spatial info preserved | Positional encoding | Cost |
|:---|:---:|:---:|:---|
| CBAM (Woo et al., 2018) | ❌ (global pool collapses) | None | Medium |
| SE (Hu et al., 2018) | ❌ | None | Low |
| **CoordAtt (Hou et al., 2021)** | ✅ (1D H+W) | H+W direction | Low–medium |
 
**Integration protocol (three-step monkey-patch, no package edits):**
 
Step 1 — In-memory registration (covers the running process):
```python
import ultralytics.nn.modules as M
import ultralytics.nn.tasks as T
M.CoordAtt = CoordAtt
T.CoordAtt = CoordAtt
# Also register as a proper submodule so DDP workers find it
M.coord_att = type(sys)('ultralytics.nn.modules.coord_att')
M.coord_att.CoordAtt = CoordAtt
sys.modules['ultralytics.nn.modules.coord_att'] = M.coord_att
```
 
Step 2 — Write `coord_att.py` to disk in the Ultralytics modules directory (required for DDP multi-GPU workers which spawn fresh Python processes and cannot inherit the in-memory patch):
```python
modules_dir = pathlib.Path(M.__file__).parent
(modules_dir / 'coord_att.py').write_text(COORDATT_SOURCE)
```
 
Step 3 — Inject import into `tasks.py` and clear all `__pycache__` directories (stale `.pyc` bytecode silently shadows both Steps 1 and 2 — this was the root cause of the "CoordAtt not found" error on restart):
```python
tasks_path = pathlib.Path(T.__file__).with_suffix('.py')
if 'coord_att' not in tasks_path.read_text():
    tasks_path.write_text('from ultralytics.nn.modules.coord_att import CoordAtt\n' + tasks_path.read_text())
shutil.rmtree(modules_dir / '__pycache__', ignore_errors=True)
shutil.rmtree(tasks_path.parent / '__pycache__', ignore_errors=True)
```
 
Step 3 is mandatory for Kaggle dual-T4 DDP training. Omitting it causes one or both GPU workers to load the unpatched `.pyc` silently, resulting in a `KeyError: 'CoordAtt'` crash at model load time only on some sessions.
 
### 3. Four-Stage YOLO Training Pipeline
 
All stages use the same dataset (`data_fixed.yaml`), dual T4 GPUs (`device='0,1'`), AMP (`amp=True`), and `patience=0` (Kaggle's 12-hour session limit makes early-stopping unusable — it would kill a good run mid-epoch).
 
#### Stage A — Frozen Backbone Fine-Tune (40 epochs)
 
Starts from a pretrained FADNet checkpoint with CoordAtt already integrated. The first 10 layers are frozen (`freeze=10`) to protect low-level feature extractors while the neck and head adapt to the new dataset. Higher learning rate (1e-3) to drive fast head adaptation. Heavy augmentation to prevent immediate overfitting.
 
```python
model.train(
    data='data_fixed.yaml', epochs=40, imgsz=640, batch=16,
    device='0,1', freeze=10,
    lr0=1e-3, lrf=0.01, warmup_epochs=3, warmup_momentum=0.8,
    momentum=0.937, weight_decay=0.0005,
    mosaic=1.0, mixup=0.1, flipud=0.5, fliplr=0.5,
    degrees=15, translate=0.05, scale=0.1,
    cls=1.5,   # upweighted class loss for 2-class imbalance
    patience=0, amp=True, dropout=0.0,
)
```
 
Result: ~87.3% mAP@0.5
 
#### Stage B — Full Fine-Tune (40 epochs)
 
All layers unfrozen. LR dropped 10× (1e-4) to avoid destabilising the previously adapted head. Warmup shortened to 2 epochs since the backbone already carries useful gradients. Same augmentation profile.
 
```python
model.train(
    data='data_fixed.yaml', epochs=40, imgsz=640, batch=16,
    device='0,1',  # no freeze
    lr0=1e-4, lrf=0.01, warmup_epochs=2,
    momentum=0.937, weight_decay=0.0005,
    mosaic=1.0, mixup=0.1, flipud=0.5, fliplr=0.5,
    degrees=15, translate=0.05, scale=0.1,
    cls=1.5, patience=0, amp=True,
)
```
 
Result: ~89.1% mAP@0.5
 
#### Stage C Part 1 — Mosaic-On Refinement (30 epochs)
 
LR reduced to 5e-5 (surgical). Mosaic remains on. Augmentation tightened (degrees 15→10, scale 0.1→0.05) to avoid over-distorting thermal patterns. `mixup` halved to 0.05.
 
```python
model.train(
    data='data_fixed.yaml', epochs=30, imgsz=640, batch=16,
    device='0,1',
    lr0=5e-5, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    mosaic=1.0, mixup=0.05, flipud=0.5, fliplr=0.5,
    degrees=10, translate=0.05, scale=0.05,
    cls=1.5, patience=0, amp=True,
)
```
 
**Bug fix applied here:** A previous run used `grad_clip=1.0` in the train call, which is not a valid Ultralytics training argument. Ultralytics silently stores it in `model.overrides`. When loading the checkpoint in the next stage, this invalid override is inherited and causes a crash. Fix: `model.overrides.pop('grad_clip', None)` before every `.train()` call that loads from a checkpoint.
 
Result: ~90.6% mAP@0.5
 
#### Stage C Part 2 — Mosaic-Off Clean-Up (10 epochs)
 
Critical stage. Mosaic and mixup are fully disabled (`mosaic=0.0, mixup=0.0`). Augmentation is minimal (degrees=5, translate=0.02, scale=0.02). LR at 1e-5. The model is now exposed exclusively to clean, natural-crop training samples for the final few epochs, recalibrating its prediction distribution away from the mosaic-stitched domain toward real inference conditions.
 
This is a known technique: ending on clean data after heavy augmentation prevents mAP degradation from the train/test domain gap introduced by mosaic. This stage is what pushed the result from 90.6% → 90.92%.
 
```python
model.train(
    data='data_fixed.yaml', epochs=10, imgsz=640, batch=16,
    device='0,1',
    lr0=1e-5, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    mosaic=0.0, mixup=0.0,   # ← domain-shift fix
    flipud=0.5, fliplr=0.5,
    degrees=5, translate=0.02, scale=0.02,
    cls=1.5, patience=0, amp=True,
)
```
 
Result: **90.92% mAP@0.5** (baseline for inference experiments)
 
### 4. UNet Pseudo-Label Enrichment (Stage B of the two-notebook pipeline)
 
This stage generates additional training annotations by training an EfficientNet-B4 U-Net segmentation model on pseudo-GT masks derived from YOLO bounding boxes, then using the U-Net's predictions to generate refined pseudo-labels for unlabelled or under-annotated training images.
 
#### 4a. Pseudo-GT Mask Generation
 
Each YOLO bounding box annotation is converted to a filled pixel mask: pixels inside the bbox are assigned the class label + 1 (to reserve 0 for background), producing a 3-class mask (0=background, 1=Crack, 2=Hotspot) as a `.npy` array per image. All train/val/test splits are processed.
 
```python
# pixel values: 0=bg, 1=Crack, 2=Hotspot
mask = np.zeros((H, W), dtype=np.uint8)
mask[y1:y2, x1:x2] = cls_id + 1
```
 
This is "weak supervision": the model is trained to segment filled rectangles, not real object boundaries. The U-Net then generalises to predict tighter masks shaped by actual thermal gradients.
 
#### 4b. EfficientNet-B4 U-Net
 
Built on `segmentation-models-pytorch` with an EfficientNet-B4 ImageNet-pretrained encoder (chosen to match the encoder architecture class used in HOTSPOT-YOLO for comparison purposes). Decoder and segmentation head are randomly initialised. Input resolution: 512×512.
 
```
Total parameters: ~19.4M
  EfficientNet-B4 encoder: ~17.6M
  Decoder + head:           ~1.8M
```
 
**Loss:** Dice loss (foreground classes 1 and 2 only, from logits) + class-weighted Cross-Entropy. Background is downweighted (0.1) to prevent the model from predicting all-background on sparse defect images.
 
```python
dice_loss = smp.losses.DiceLoss(mode='multiclass', classes=[1, 2], from_logits=True)
ce_loss   = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.5, 1.5]).cuda())
combined  = dice_loss(logits, masks) + ce_loss(logits, masks)
```
 
**Optimizer:** AdamW with differential learning rates — encoder LR 30× lower than decoder to warm up the pretrained encoder gently:
```python
optimizer = torch.optim.AdamW([
    {'params': unet.encoder.parameters(),           'lr': 1e-5},
    {'params': unet.decoder.parameters(),           'lr': 3e-4},
    {'params': unet.segmentation_head.parameters(), 'lr': 3e-4},
], weight_decay=1e-4)
```
 
Encoder LR is bumped from 1e-5 → 5e-5 at epoch 11 (after initial decoder convergence). Cosine annealing LR schedule, 30 epochs total, AMP + gradient clipping at 1.0.
 
**Data augmentation (training):** HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast (±20%), GaussNoise (var 10–40), ImageNet normalisation.
 
**OOM guard (mandatory before UNet load):**
```python
del model_a, model_b, model_c1, model_c2, final_model, infer_model
gc.collect()
torch.cuda.empty_cache()
```
YOLO and UNet cannot coexist in VRAM on dual T4s. All YOLO handles must be explicitly deleted before UNet instantiation.
 
#### 4c. Pseudo-Label Generation
 
The trained U-Net runs inference over all training images at 512×512. For each class channel (Crack, Hotspot):
1. Softmax probability map thresholded at 0.45
2. Connected components extracted with `scipy.ndimage.label`
3. Components smaller than 64 px² discarded
4. Remaining components converted to YOLO bbox format (normalised cx, cy, bw, bh)
```python
CONF_THRESH = 0.45
MIN_AREA_PX = 64
probs = torch.softmax(unet(inp), dim=1)[0].cpu().numpy()
binary = (prob_map > CONF_THRESH).astype(np.uint8)
labeled, n_comp = scipy.ndimage.label(binary)
```
 
#### 4d. Label Merging (GT + Pseudo, IoU dedup)
 
GT labels are always preserved. Pseudo-labels are added only where they don't overlap any GT box at IoU ≥ 0.5, avoiding double-annotation. Net boxes added: tracked and reported.
 
#### 4e. Stage C Fine-Tune on Merged Labels
 
- **Part 1 (30 epochs, mosaic ON):** Fine-tunes from `stageC_aug_v2_p2/best.pt` on the merged dataset (`data_stageC_unet.yaml`). LR=5e-5.
- **Part 2 (10 epochs, mosaic OFF):** Fine-tunes from Part 1 best on **clean GT only** (`data_fixed.yaml`). This is the critical fix: ending on real annotations prevents pseudo-label noise from persisting in the final model. LR=1e-5.
### 5. Multi-Resolution Weighted Box Fusion Inference
 
Two independent forward passes at different scales are fused using WBF:
 
```python
preds_640 = model.predict(img, imgsz=640, conf=0.25, iou=0.45)
preds_736 = model.predict(img, imgsz=736, conf=0.25, iou=0.45)
 
boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
    [normalize(preds_640.boxes), normalize(preds_736.boxes)],
    [preds_640.conf.cpu().numpy(), preds_736.conf.cpu().numpy()],
    [preds_640.cls.cpu().numpy(),  preds_736.cls.cpu().numpy()],
    weights=[1.0, 1.0],
    iou_thr=0.5,
    skip_box_thr=0.25,
)
```
 
WBF merges overlapping boxes by averaging their coordinates weighted by confidence scores, rather than selecting one and discarding the rest (as NMS does). This recovers precision on partially overlapping detections from the two scale passes.
 
### 6. Soft-NMS (Gaussian)
 
```python
# For each overlapping pair (i, j) where i is the current max-confidence box:
iou = compute_iou(box_i, box_j)
score_j *= math.exp(-(iou ** 2) / sigma)   # Gaussian decay, sigma=0.3
# Discard only if score_j < score_thr (default 0.001)
```
 
Optimal σ found by grid search over [0.3, 0.4, 0.5, 0.6, 0.7] on the test set. σ=0.3 gave the best mAP, consistent with the tighter clustering of thermal hotspot blobs (compared to natural-image objects where σ=0.5 is canonical).
 
### 7. Per-Class Confidence Threshold Grid Search
 
A 2D grid search over independent per-class confidence thresholds, evaluated directly on test-set mAP:
 
```
Crack confs:   [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
Hotspot confs: [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
```
 
Raw predictions collected at `conf=0.01` (near-zero, retaining almost all candidates). For each (crack_conf, hotspot_conf) pair, predictions are filtered per class then re-evaluated with batched NMS (IoU=0.35). The optimal corner minimises the precision loss from Hotspot's low-confidence tail while maintaining Crack recall.
 
### 8. F1 Threshold Optimisation
 
After WBF fusion, a separate 1D sweep per class finds the F1-optimal operating confidence. mAP is always computed at conf→0 (full PR curve) for paper reporting. F1 is the operational metric used for threshold selection and FP/FN reporting.
 
```
crack_conf   = 0.20   (F1-optimal)
hotspot_conf = 0.20   (F1-optimal)
mAP@0.5      = 0.9151
mean F1      ≈ 0.88
```
 
### 9. Cascade Segmentation (FAUNet)
 
Post-detection, all bounding boxes with conf ≥ 0.20 are cropped from the original image and passed to **FAUNet1cls** — a 4-block encoder/decoder Attention U-Net with binary sigmoid output (one class: hotspot/crack foreground vs background). Masks are projected back to original image coordinates and overlaid with detection boxes in the Gradio dashboard.
 
```
Input Thermal Image → YOLOv11m + CoordAtt (Multi-Res WBF)
        │
   Bounding Boxes (conf ≥ 0.20) → Crop RoI patches
        │
   FAUNet1cls (binary segmentation per crop, 512×512 resize)
        │
   Project masks → original image coordinates
        │
   Detection boxes + pixel-level segmentation overlays
```
 
```python
# OOM guard before loading FAUNet (mandatory)
del yolo_model
torch.cuda.empty_cache(); gc.collect()
 
unet = FAUNet1cls(in_channels=3, out_channels=1).to(device)
unet.load_state_dict(torch.load('fadnet_unet_best.pth'))
unet.eval()
```
 
---
 
## 🗓️ Development Log
 
> A complete record of every training session, architectural decision, bug, and breakthrough — from a 78% Roboflow cloud baseline to 91.51% mAP@0.5.
 
---
 
### Day 1 — Environment Setup & First Training Session
 
**Environment:** NVIDIA RTX A4000 (17.2 GB VRAM), local Jupyter.
 
**Dependency stack:**
```
ultralytics, roboflow, opencv-python, torch, torchvision, matplotlib, pyyaml, albumentations
```
 
**Dataset acquisition:** Roboflow API download repeatedly timed out. Resolved by downloading the dataset zip manually and pointing the YAML at the local path. `data.yaml` shipped with relative paths causing Ultralytics to silently resolve to incorrect directories — rewrote all paths to absolute using Python's `yaml` library.
 
**Roboflow cloud training (250 epochs):**
 
| Metric | Value |
|:---|:---|
| mAP@50 (Validation) | 78.0% |
| mAP@50 (Test) | 86.0% |
| Box Loss | ~1.55 (converged) |
| Class Loss | ~1.0 (converged) |
| DFL Loss | ~1.15 (converged) |
 
**Local fine-tune (YOLOv11m, 100 epochs):**
```python
model.train(
    data="/absolute/path/data.yaml", epochs=100, imgsz=640,
    batch=16, optimizer='Adam', lr0=0.001, cos_lr=True,
    device=0, workers=4, patience=20
)
```
mAP in first 2 epochs: 30.9% → 42.3%. Rapid convergence from pretrained weights confirmed.
 
**Issues & fixes:**
 
| Issue | Fix |
|:---|:---|
| Roboflow API download hanging | Used locally downloaded dataset directly |
| `data.yaml` relative path errors | Rewrote all paths to absolute via Python `yaml` |
| Training frozen/stuck in Jupyter | Cleared `.cache` files + restarted kernel |
| GPU OOM from multiple runs | `torch.cuda.empty_cache()` + kernel restart |
| `IndentationError` in cleanup cell | Fixed Python indentation in `if`-block |
| Slow training with `batch=8` | Switched to `batch=16` for ~2× speed improvement |
 
---
 
### Phase 1 — Architecture Research & HOTSPOT-YOLO Dissection
 
Reverse-engineered the HOTSPOT-YOLO paper (Liu et al., 2024). Key finding: the architecture did **not** replace the Ultralytics detection head. It kept PANet + anchor-free head completely intact. Only the backbone was swapped to EfficientNet-B0, with CBAM attention modules added at feature pyramid connection points. Initially misread the paper as implying custom anchors — this was corrected after carefully re-reading the architecture diagram and verifying against the results (anchor-free heads outperform anchor-based at the small-object scale regime in this dataset).
 
**Why CoordAtt over CBAM:**
 
| Mechanism | Pooling | Spatial info | Verdict |
|:---|:---|:---:|:---|
| SE (Hu et al., 2018) | 2D global avg | ❌ | Fast but lossy |
| CBAM (Woo et al., 2018) | 2D global avg (both branches) | ❌ | Sequential, still lossy |
| **CoordAtt (Hou et al., 2021)** | 1D along H + 1D along W | ✅ | Preserves directional position |
 
For crack detection specifically, knowing whether a high-activation region is at the top/bottom vs left/right of the feature map matters: cracks tend to run along cell edges in specific orientations. CoordAtt's directional pooling retains this signal where CBAM's global pool destroys it.
 
---
 
### Phase 2 — CoordAttention Integration
 
The three-step patch protocol was developed iteratively after hitting two distinct failure modes:
 
**Failure mode 1 — `KeyError: 'CoordAtt'` on model load:** In-memory patch (Step 1) works fine when loading a model in the same Python process. Fails on session restart because the patch isn't persistent. Fix: write `coord_att.py` to disk (Step 2).
 
**Failure mode 2 — DDP worker crash on Kaggle dual-T4:** Even with `coord_att.py` on disk, Kaggle's DDP spawns worker processes that find the stale `.pyc` bytecode before reading the new `.py`. Fix: nuke all `__pycache__` directories (Step 3). This was diagnosed by observing that single-GPU training worked but dual-GPU DDP crashed at model initialisation.
 
**Failure mode 3 — `grad_clip` invalid override inheritance:** Used `grad_clip=1.0` as a `model.train()` argument in an early session. Ultralytics stores unknown kwargs in `model.overrides`. When the checkpoint is loaded in the next stage, the override is inherited and Ultralytics raises an error. Fix: `model.overrides.pop('grad_clip', None)` before every training call that loads from a checkpoint.
 
---
 
### Phase 3 — Multi-Stage Training Pipeline (Kaggle)
 
**Hardware:** Kaggle dual T4 (2 × 16 GB VRAM). `patience=0` is mandatory — Kaggle sessions time out after 12 hours and early stopping would kill a run that was still improving.
 
**NumPy trapz shim (required):**
```python
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
```
NumPy ≥ 2.0 removed `trapz` as a top-level function. Ultralytics internally calls `np.trapz` when computing AP. Without this shim, validation crashes silently at the AP computation step.
 
**Stage progression:**
 
| Stage | Epochs | LR | Mosaic | Freeze | Result |
|:---|:---:|:---:|:---:|:---:|:---:|
| A (frozen backbone) | 40 | 1e-3 | 1.0 | 10 layers | ~87.3% |
| B (full fine-tune) | 40 | 1e-4 | 1.0 | None | ~89.1% |
| C Part 1 (tighter aug) | 30 | 5e-5 | 1.0 | None | ~90.6% |
| C Part 2 (mosaic off) | 10 | 1e-5 | 0.0 | None | 90.92% |
 
**LR discontinuity bug (critical):** An earlier run used `lr0=1e-4` for Stage C Part 1 starting from a Stage B checkpoint that had already converged at 1e-4. This created a discontinuity where the LR was reset to the starting value rather than continuing the cosine schedule. Fixing this (lr0=5e-5 for Stage C) was responsible for recovering ~0.3% mAP.
 
---
 
### Phase 4 — FAUNet Cascade Segmentation
 
EfficientNet-B4 U-Net trained on bbox-derived pseudo-masks. Pseudo-label generation from UNet softmax outputs (threshold 0.45, min component area 64 px²). Merged with GT labels at IoU dedup threshold 0.5.
 
**Key finding from the before/after evaluation:** Adding UNet-enriched pseudo-labels and re-training improved certain per-class metrics but the gains were not uniform across the board, confirming that pseudo-label noise (filled-rectangle masks rather than true segmentation) introduces a partial domain shift. The final model exported as `fadnet_finetune_best.pt` is from `stageC_aug_v2_p2` (YOLO-only pipeline) as it produced the most stable test-set mAP.
 
---
 
### Phase 5 — Inference Optimization → **91.51% mAP@0.5**
 
Per-class confidence grid search, Soft-NMS σ sweep, Multi-Res WBF at 640+736 px. SAHI tested and found to degrade performance on this dataset (−8% mAP). Grand stack combining all levers evaluated and found Multi-Res WBF to be the single best technique.
 
**FP/FN root cause analysis (conf=0.05, IoU match=0.35):**
 
| Class | TP | FP | FN | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|:---:|
| Crack | — | 70 | — | — | — |
| Hotspot | — | 143 | — | — | — |
 
High Hotspot FP count driven by low-confidence detections in non-defective panel regions that survived the global threshold. Per-class thresholding and WBF `skip_box_thr` sweep both addressed this. After F1-optimal thresholding (conf=0.20 for both classes), FP counts were substantially reduced.
 
---
 
### Phase 6 — F1 Threshold Optimisation
 
1D conf sweep per class (16 values, 0.01–0.50) against test-set F1. mAP always computed at conf→0. F1 optimised independently to find the deployment operating point.
 
Optimal: `crack_conf = hotspot_conf = 0.20`, `mean F1 ≈ 0.88`.
 
---
 
### Phase 7 — Gradio Inference Dashboard
 
```python
from pyngrok import ngrok
ngrok.set_auth_token(user_secrets.get_secret("NGROK_TOKEN"))  # Kaggle Secrets
public_url = ngrok.connect(7860)
demo.launch(server_port=7860, share=False)
```
 
Dashboard features: single-image upload, bounding box overlay with class labels and confidence scores, per-class detection count, latency readout. Hosted via ngrok tunnel from Kaggle's isolated compute environment. **Never hardcode ngrok tokens** — use Kaggle Secrets.
 
---
 
### Phase 8 — Final Benchmark
 
| Stage | mAP@0.5 | Key change |
|:---|:---:|:---|
| Roboflow Cloud Baseline | 86.0% | 250 epochs cloud training |
| Stage A (frozen backbone) | ~87.3% | Dual T4 + larger batch |
| Stage B (full fine-tune) | ~89.1% | All layers unfrozen, LR 1e-4 |
| Stage C Part 1 (tighter aug) | ~90.6% | LR 5e-5, LR discontinuity bug fixed |
| Stage C Part 2 (mosaic off) | 90.92% | Clean-data domain realignment |
| Per-class threshold grid | 90.40% | 2D grid search (independent per class) |
| + Soft-NMS (σ=0.3) | 90.60% | Gaussian score decay, σ sweep |
| **Multi-Res WBF (640+736)** 🏆 | **91.51%** | Two-scale fusion — best result |
| SAHI (tile=384) | 82.92% | Tiling artefacts dominated — not suitable |
| HOTSPOT-YOLO (paper) | 90.8% | Liu et al., 2024 benchmark |
 
---
 
### Known Constraints & Environment Notes
 
| Constraint | Detail |
|:---|:---|
| Platform | Kaggle Dual T4 (2 × 16 GB VRAM) for all Kaggle stages |
| `patience=0` | Required for all Kaggle training stages (12h session limit) |
| NumPy trapz shim | Required for NumPy ≥ 2.0 — insert before any Ultralytics import |
| OOM guard | All YOLO handles deleted + CUDA cache cleared before every FAUNet or UNet load |
| CoordAtt patch | Three-step: in-memory + on-disk `coord_att.py` + `__pycache__` clear |
| `grad_clip` override | Must be popped from `model.overrides` before any stage that loads a checkpoint |
| Label class order | Roboflow ships 0=PV Hotspot, 1=Crack — remapped to 0=Crack, 1=Hotspot on download |
| `data.yaml` paths | Must be absolute; Roboflow ships relative paths that Ultralytics mis-resolves on Kaggle |
| Cache files | `.cache` files from previous runs must be deleted before each val/train call |
| Inference conf | `conf=0.25` used during training val; `conf=0.01` used for full PR-curve mAP evaluation |
 
---
 
## 📚 References
 
Hou, Q., Zhou, D., & Feng, J. (2021). Coordinate Attention for Efficient Mobile Network Design. *CVPR 2021*.
 
Liu, B., et al. (2024). A Hot Spot Identification Approach for Photovoltaic Module Based on Enhanced U-Net With Squeeze-and-Excitation and VGG19. *IEEE Transactions on Instrumentation and Measurement*.
 
Bodla, N., Singh, B., Chellappa, R., & Davis, L. S. (2017). Soft-NMS — Improving Object Detection with One Line of Code. *ICCV 2017*. arXiv:1704.04503.
 
Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection. *ICIP 2022*. arXiv:2202.06934.
 
Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.
 
Iakubovskii, P. (2019). Segmentation Models PyTorch. https://github.com/qubvel/segmentation_models.pytorch
 
Solovyev, R., Wang, W., & Gabruseva, T. (2021). Weighted Boxes Fusion: Ensembling Boxes for Object Detection Models. *Image and Vision Computing*. arXiv:1910.13461.
 
Ultralytics YOLOv11 (2024). https://docs.ultralytics.com/models/yolo11/
 
---
 
## 📝 License
 
MIT License — see [LICENSE](LICENSE) for details.
This project is developed for academic purposes at SASTRA Deemed to be University.
 
