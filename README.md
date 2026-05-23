# FADNet — Fault-Aware Detection Network for Photovoltaic Thermal Imaging
 
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-green)
![mAP@0.5](https://img.shields.io/badge/mAP%400.5-91.51%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-success)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20Dual%20T4-informational)
![Paper](https://img.shields.io/badge/Paper-Coming_Soon-lightgrey)
 
---
 
## 📌 Abstract
 
Photovoltaic (PV) module hotspots and cracks dramatically reduce energy yield and pose long-term safety risks in solar installations. Conventional fault detection methods — I-V curve analysis, image-level classification, and naïve bounding box detectors — struggle to localise precise fault boundaries, especially for thermally small or spatially clustered defects.
 
**FADNet** is a fault-aware detection network built on **YOLOv11m** with a custom **CoordAttention (CoordAtt)** mechanism injected into the backbone. CoordAtt replaces standard 2D global pooling with two 1D pooling operations along H and W axes independently, producing channel-attention maps that retain directional positional information — a property especially important for detecting elongated crack patterns in thermal imagery.
 
The training pipeline is fully progressive: a four-stage YOLO curriculum (frozen-backbone → full fine-tune → mosaic-on refinement → mosaic-off clean-up) is followed by a UNet-based pseudo-label enrichment stage. Inference is pushed further via Multi-Resolution Weighted Box Fusion (fusing predictions at 640 px and 736 px) plus Soft-NMS suppression and per-class confidence thresholding.
 
**Final result: mAP@0.5 = 91.51%** on the Thermal-H&C test split (Hotspot AP = 94.15%, Crack AP = 88.86%), exceeding the HOTSPOT-YOLO paper's reported 90.8%.
 
---
 
## 🏗️ Architecture
 
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
 
| Technique | mAP@0.5 | Hotspot AP | Crack AP | Δ vs Baseline |
|:---|:---:|:---:|:---:|:---:|
| Baseline WBF (Stage C2) | 90.92% | — | — | — |
| Per-class conf threshold | 90.40% | — | — | −0.52% |
| + Soft-NMS (σ=0.3) | 90.60% | — | — | −0.32% |
| **Multi-Res WBF (640+736)** 🏆 | **91.51%** | **94.15%** | **88.86%** | **+0.59%** |
| SAHI (tile=384, overlap=0.4) | 82.92% | — | — | −8.00% |
| HOTSPOT-YOLO (Liu et al. 2024) | 90.8% | — | — | — |
 
```
crack_conf    = 0.20
hotspot_conf  = 0.20
mAP@0.5       = 0.9151   ← paper metric (full PR curve, conf → 0)
mean F1       ≈ 0.88     ← operational metric at above thresholds
```
 
---
 
## 📂 Repository Structure
 
```
FADNet/
├── app.py                        ← Gradio inference dashboard (ngrok-tunnelled)
├── requirements.txt              ← Python dependencies
├── fadnet_base.ipynb             ← Full training pipeline: Stage A→B→C + UNet + diagnostics (Kaggle)
├── fadnet_trainingv1.ipynb       ← Advanced inference push: Multi-Res WBF, SAHI, Soft-NMS, F1 sweep (Kaggle)
├── checkpoints/                  ← Model weights (see note below)
└── working/
    ├── fadnet_metrics_dashboard.png
    ├── fadnet_advanced_push.png
    ├── fadnet_bbox_quality.png
    ├── fadnet_live_inference.png
    ├── fadnet_result_grid.png
    ├── f1_optimal_curves.png
    └── perclass_thresh_heatmap.png
```
 
> **Model weights** are not hosted in this repository due to file size constraints. Available on request — reach out via [LinkedIn](https://linkedin.com/in/ashvathram-b-59b762284).
 
> **Notebook execution order:**
> 1. `fadnet_base.ipynb` — Run Cell 1 (CoordAtt patch) every session. Then Cells 2–30 in order.
> 2. `fadnet_trainingv1.ipynb` — Inference-only. Run Cell 1 (CoordAtt patch) then Cells 2–20 in order.
 
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
 
**To reproduce training on Kaggle:**
- Upload `fadnet_base.ipynb` to a Kaggle notebook with GPU accelerator = **T4 × 2**
- Add your Roboflow API key as a Kaggle Secret named `ROBOFLOW_API_KEY`
- Run all cells in order
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
 
<details>
<summary><b>1. Dataset</b></summary>
The dataset was pulled from Roboflow workspace, using the `yolov11` export format.
 
| Split | Images |
|:---|:---:|
| Train | 1,923 |
| Validation | 205 |
| Test | ~200 |
 
**Critical label remap:** The raw Roboflow download ships class 0 = `PV Hotspot`, class 1 = `Crack`. FADNet's training inverts this to class 0 = `Crack`, class 1 = `Hotspot`. All label files are remapped on download using a single-pass script that flips `0 → 1` and `1 → 0` in every `.txt` annotation.
 
**data.yaml fix:** Roboflow ships `data.yaml` with relative paths which Ultralytics resolves incorrectly in Kaggle's working directory. All paths are rewritten to absolute on download.
 
</details>
<details>
<summary><b>2. CoordAttention Backbone</b></summary>
CoordAtt (Hou et al., CVPR 2021) decomposes 2D global average pooling into two separate 1D pooling operations — one along H, one along W — producing attention maps that encode both channel dependencies and spatial directional context.
 
```
Input Feature Map: (B, C, H, W)
        │
        ├─→ Mean pool along W axis → (B, C, H, 1)   [vertical positional encoding]
        └─→ Mean pool along H axis → (B, C, 1, W)   [horizontal positional encoding]
                 permute → (B, C, W, 1)
        │
        Concatenate along H dim → (B, C, H+W, 1)
        │
        Conv2d(C → C//r, 1×1) → BN → h_swish activation
        │
        Split along H dim:
          ├─→ (B, C//r, H, 1) → Conv2d → Sigmoid → attention_h
          └─→ (B, C//r, W, 1) → Conv2d → Sigmoid → attention_w
        │
        Output = Input × attention_h × attention_w
```
 
**Why CoordAtt over CBAM:**
 
| Mechanism | Spatial info preserved | Positional encoding | Cost |
|:---|:---:|:---:|:---|
| CBAM (Woo et al., 2018) | ❌ (global pool collapses) | None | Medium |
| SE (Hu et al., 2018) | ❌ | None | Low |
| **CoordAtt (Hou et al., 2021)** | ✅ (1D H+W) | H+W direction | Low–medium |
 
**Integration protocol (three-step monkey-patch):**
 
Step 1 — In-memory registration:
```python
import ultralytics.nn.modules as M
import ultralytics.nn.tasks as T
M.CoordAtt = CoordAtt
T.CoordAtt = CoordAtt
M.coord_att = type(sys)('ultralytics.nn.modules.coord_att')
M.coord_att.CoordAtt = CoordAtt
sys.modules['ultralytics.nn.modules.coord_att'] = M.coord_att
```
 
Step 2 — Write to disk (required for DDP multi-GPU workers):
```python
modules_dir = pathlib.Path(M.__file__).parent
(modules_dir / 'coord_att.py').write_text(COORDATT_SOURCE)
```
 
Step 3 — Clear `__pycache__` (stale `.pyc` silently shadows Steps 1 and 2):
```python
tasks_path = pathlib.Path(T.__file__).with_suffix('.py')
if 'coord_att' not in tasks_path.read_text():
    tasks_path.write_text('from ultralytics.nn.modules.coord_att import CoordAtt\n' + tasks_path.read_text())
shutil.rmtree(modules_dir / '__pycache__', ignore_errors=True)
shutil.rmtree(tasks_path.parent / '__pycache__', ignore_errors=True)
```
 
</details>
<details>
<summary><b>3. Four-Stage YOLO Training Pipeline</b></summary>
All stages: dual T4 GPUs (`device='0,1'`), AMP (`amp=True`), `patience=0`.
 
| Stage | Epochs | LR | Mosaic | Freeze | Result |
|:---|:---:|:---:|:---:|:---:|:---:|
| A (frozen backbone) | 40 | 1e-3 | 1.0 | 10 layers | ~87.3% |
| B (full fine-tune) | 40 | 1e-4 | 1.0 | None | ~89.1% |
| C Part 1 (tighter aug) | 30 | 5e-5 | 1.0 | None | ~90.6% |
| C Part 2 (mosaic off) | 10 | 1e-5 | 0.0 | None | 90.92% |
 
**LR discontinuity bug (critical):** An earlier run used `lr0=1e-4` for Stage C Part 1 starting from a Stage B checkpoint that had already converged at 1e-4 — resetting the LR rather than continuing the cosine schedule. Fixing this (lr0=5e-5) recovered ~0.3% mAP.
 
**NumPy trapz shim (required for NumPy ≥ 2.0):**
```python
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
```
 
</details>
<details>
<summary><b>4. UNet Pseudo-Label Enrichment</b></summary>
An EfficientNet-B4 U-Net (segmentation-models-pytorch) is trained on bbox-derived pseudo-GT masks, then used to generate refined pseudo-labels for training images.
 
- Pseudo-GT: filled pixel masks from YOLO bbox annotations (3-class: background, Crack, Hotspot)
- Loss: Dice (foreground only) + class-weighted CrossEntropy (background weight=0.1)
- Optimizer: AdamW with differential LRs (encoder 1e-5, decoder 3e-4)
- Pseudo-label threshold: softmax prob > 0.45, min component area 64 px²
- GT + pseudo deduplication: IoU ≥ 0.5 → pseudo label discarded
**OOM guard (mandatory before UNet load):**
```python
del model_a, model_b, model_c1, model_c2, final_model, infer_model
gc.collect()
torch.cuda.empty_cache()
```
 
</details>
<details>
<summary><b>5. Multi-Resolution WBF + Soft-NMS + Per-Class Thresholding</b></summary>
**Multi-Res WBF:** Two independent forward passes at 640 px and 736 px fused with Weighted Box Fusion (IoU threshold 0.5). Each resolution resolves differently-sized defects — best single technique, +0.59% mAP at zero retraining cost.
 
**Soft-NMS (Gaussian):** `score_j *= exp(−IoU² / σ)`, σ=0.3. Prevents suppression of adjacent genuine hotspots on neighbouring PV cells.
 
**Per-class conf grid search:** 2D sweep over independent Crack/Hotspot thresholds at conf=0.01 base, re-evaluated with batched NMS per pair. Optimal: `crack_conf = hotspot_conf = 0.20`.
 
**SAHI:** Tested and degraded mAP by 8% — tiling at 384 px created partial-box edge artefacts on defects that are not unusually small relative to image size.
 
</details>
<details>
<summary><b>6. Cascade Segmentation (FAUNet1cls)</b></summary>
Post-detection RoIs (conf ≥ 0.20) are cropped and passed to FAUNet1cls — a 4-block encoder/decoder Attention U-Net with binary sigmoid output. Masks are projected back to original image coordinates and overlaid in the Gradio dashboard.
 
```
Input → YOLOv11m + CoordAtt (Multi-Res WBF)
      → Crop RoIs (conf ≥ 0.20)
      → FAUNet1cls (512×512)
      → Project masks → original coordinates
      → Detection boxes + segmentation overlays
```
 
```python
del yolo_model
torch.cuda.empty_cache(); gc.collect()
unet = FAUNet1cls(in_channels=3, out_channels=1).to(device)
unet.load_state_dict(torch.load('fadnet_unet_best.pth'))
unet.eval()
```
 
</details>
---
 
## 🗓️ Development Log
 
<details>
<summary><b>Click to expand full development log (86% → 91.51%)</b></summary>
 
### Phase 1 — Architecture Research & HOTSPOT-YOLO Dissection
 
Reverse-engineered Liu et al. (2024). Key finding: HOTSPOT-YOLO kept PANet + anchor-free head intact — only the backbone was swapped to EfficientNet-B0 with CBAM at FPN connection points. Initially misread as implying custom anchors — corrected after re-reading the architecture diagram.
 
### Phase 2 — CoordAttention Integration
 
Three failure modes encountered and fixed:
- `KeyError: 'CoordAtt'` on session restart → write `coord_att.py` to disk
- DDP worker crash on dual-T4 → nuke `__pycache__` (stale `.pyc` shadowed the new `.py`)
- `grad_clip` invalid override inheritance → `model.overrides.pop('grad_clip', None)` before every checkpoint load
### Phase 3 — Multi-Stage Training (Kaggle Dual T4)
 
Stage A → B → C1 → C2 progressive pipeline. LR discontinuity bug between stages diagnosed and fixed (+0.3% mAP). `patience=0` mandatory for Kaggle 12h session limit.
 
### Phase 4 — FAUNet Cascade
 
EfficientNet-B4 U-Net trained on bbox-derived pseudo-masks. Key finding: pseudo-label noise from filled-rectangle masks introduces partial domain shift — final deployed model is YOLO-only pipeline (`stageC_aug_v2_p2/best.pt`).
 
### Phase 5 — Inference Optimisation → 91.51%
 
Multi-Res WBF best single technique (+0.59%). SAHI degraded performance (−8%). Per-class thresholding + Soft-NMS provided marginal gains. Grand stack evaluated — WBF alone is optimal.
 
### Phase 6 — F1 Threshold Optimisation
 
1D conf sweep per class. Optimal: `crack_conf = hotspot_conf = 0.20`, `mean F1 ≈ 0.88`.
 
### Final Benchmark
 
| Stage | mAP@0.5 | Key change |
|:---|:---:|:---|
| Roboflow Cloud Baseline | 86.0% | 250 epochs cloud training |
| Stage A | ~87.3% | Dual T4 + larger batch |
| Stage B | ~89.1% | All layers unfrozen |
| Stage C Part 1 | ~90.6% | LR discontinuity bug fixed |
| Stage C Part 2 | 90.92% | Mosaic-off domain realignment |
| **Multi-Res WBF** 🏆 | **91.51%** | Two-scale fusion |
| HOTSPOT-YOLO (paper) | 90.8% | Liu et al., 2024 |
 
</details>
---
 
## ⚠️ Known Constraints
 
| Constraint | Detail |
|:---|:---|
| Platform | Kaggle Dual T4 (2 × 16 GB VRAM) |
| `patience=0` | Required for all Kaggle training stages |
| NumPy trapz shim | Required for NumPy ≥ 2.0 |
| OOM guard | Delete all YOLO handles + clear CUDA cache before FAUNet load |
| CoordAtt patch | Three-step: in-memory + on-disk + `__pycache__` clear |
| `grad_clip` override | Pop from `model.overrides` before every checkpoint load |
| Label class order | Roboflow ships 0=Hotspot, 1=Crack — remapped on download |
| `data.yaml` paths | Must be absolute on Kaggle |
| Inference conf | `conf=0.25` for training val; `conf=0.01` for full PR-curve mAP |
 
---
 
## 📚 References
 
Hou, Q., Zhou, D., & Feng, J. (2021). Coordinate Attention for Efficient Mobile Network Design. *CVPR 2021*.
 
Liu, B., et al. (2024). A Hot Spot Identification Approach for Photovoltaic Module Based on Enhanced U-Net With Squeeze-and-Excitation and VGG19. *IEEE Transactions on Instrumentation and Measurement*.
 
Bodla, N., Singh, B., Chellappa, R., & Davis, L. S. (2017). Soft-NMS — Improving Object Detection with One Line of Code. *ICCV 2017*. arXiv:1704.04503.
 
Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection. *ICIP 2022*. arXiv:2202.06934.
 
Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.
 
Solovyev, R., Wang, W., & Gabruseva, T. (2021). Weighted Boxes Fusion. *Image and Vision Computing*. arXiv:1910.13461.
 
Ultralytics YOLOv11 (2024). https://docs.ultralytics.com/models/yolo11/
 
---
 
## 📝 License
 
MIT License — see [LICENSE](LICENSE) for details.
