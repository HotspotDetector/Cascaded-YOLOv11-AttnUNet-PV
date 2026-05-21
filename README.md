# A Cascaded YOLOv11 and Attention-Driven U-Net Framework for Autonomous Photovoltaic Hotspot Diagnosis
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-green)
![Status](https://img.shields.io/badge/Status-In_Development-yellow)

> **Course:** CSE300 Mini Project (Batch M46)
> **Institution:** SASTRA Deemed to be University, Srinivasa Ramanujan Center

## 📌 Abstract
In solar power systems, Photovoltaic (PV) module hot spots dramatically lower efficiency and pose safety hazards. Traditional methods like I-V curve analysis, image-level classification, and simple bounding box detection often struggle to pinpoint the precise location and size of these faults.

This project implements a **Cascaded Deep Learning Framework** that combines the speed of **YOLOv11** with the precision of an **Attention-Driven U-Net**.

1.  **Localization:** YOLOv11 rapidly locates possible hot spot regions and eliminates unimportant background information (roof tiles, vegetation).
2.  **Segmentation:** The cropped regions are fed into a U-Net segmentation network to identify hot spots at the pixel level.

This hybrid approach allows for the precise measurement of fault size, shape, and severity, outperforming standalone detection models in terms of pixel accuracy and Mean Intersection-Over-Union (MIoU).



## 🛠️ Methodology
Our framework addresses the trade-off between real-time speed and segmentation accuracy by using a two-stage pipeline:

### 1. Image Pre-processing
To handle the limited labeled data typical of real-world PV inspections and increase robustness, we apply:
* **Gaussian Blurring:** To reduce thermal noise.
* **Image Sharpening:** To improve edge definition and temperature contrast.

### 2. The Cascade Architecture
* **Stage 1 (Global Detection):** A **YOLOv11** model scans the full thermal image to detect Regions of Interest (RoIs).
* **Stage 2 (Local Segmentation):** The detected RoIs are cropped and passed to an **Attention-Driven U-Net**. This network generates a binary mask to segment the exact shape of the hot spot.

## 📂 Dataset
The model is trained on **Infrared Thermal Images** of photovoltaic modules. The dataset includes classes for:
* `Hotspot`
* `Defective Module`
* `Diode Failure`

---

## 🗓️ Day 1 — Thursday, February 19, 2026 | Local Implementation & Training Session

> This section documents the hands-on implementation work carried out during the first active development session.

### ✅ What We Did Today

#### 1. 🔧 Environment Setup
- Installed all required dependencies: `ultralytics`, `roboflow`, `opencv-python`, `torch`, `matplotlib`
- Verified GPU availability — **NVIDIA RTX A4000 (17.2 GB VRAM)** confirmed with CUDA

#### 2. 📦 Dataset Acquisition
- Connected to **Roboflow** workspace (`your-workspace`) and downloaded the `solar-thermal-hotspot` dataset (Version 1)
- Dataset contains **~2000 thermal PV images** across train/valid/test splits
- Local dataset path configured at:
  ```
  your-local-path\Dataset\solar-thermal-hotspot.v1-yolov11.yolov11\
  ```
- Fixed relative paths in `data.yaml` to absolute local paths for Jupyter compatibility

#### 3. ☁️ Roboflow Cloud Training (Completed)
- Trained YOLOv11 model directly on the Roboflow platform for **250 epochs**
- Results:

| Metric | Value |
|:---:|:---:|
| mAP@50 (Validation Set) | **78.0%** |
| mAP@50 (Test Set) | **86.0%** |
| Box Loss | Converged ~1.55 |
| Class Loss | Converged ~1.0 |
| Object Loss | Converged ~1.15 |

- All three losses converged smoothly with **no overfitting** observed across 250 epochs

#### 4. 💻 Local Training in Jupyter Notebook
- Set up end-to-end training pipeline locally using **Ultralytics YOLOv11**
- Fine-tuned pretrained `yolo11m.pt` weights on the thermal PV dataset
- Training configuration used:

```python
model.train(
    data      = "data.yaml",
    epochs    = 100,
    imgsz     = 640,
    batch     = 16,
    optimizer = 'Adam',
    lr0       = 0.001,
    cos_lr    = True,
    device    = 0,        # NVIDIA RTX A4000
    workers   = 4
)
```

#### 5. 📊 Training Observations
- GPU Memory usage: **~9.3 GB / 17.2 GB** during training
- Training speed: **~2.7 iterations/second**
- Each epoch completed in approximately **~50 seconds**
- mAP improved rapidly in early epochs: **30.9% → 42.3%** within just 2 epochs

### 🔍 Issues Faced & How We Fixed Them

| Issue | Fix Applied |
|:---|:---|
| Roboflow API download hanging | Used locally downloaded dataset folder directly |
| `data.yaml` relative path errors | Rewrote paths to absolute using Python `yaml` library |
| Training frozen/stuck in Jupyter | Cleared `.cache` files + restarted Jupyter kernel |
| GPU memory overflow from multiple runs | `torch.cuda.empty_cache()` + kernel restart |
| `IndentationError` in cleanup cell | Fixed Python indentation in if-block |
| Slow training with `batch=8` | Switched back to `batch=16` for 2x speed improvement |

### 📁 Local Project Structure (End of Day 1)
```
your-local-path\
├── Dataset\
│   └── solar-thermal-hotspot.v1-yolov11.yolov11\
│       ├── train\images\        ← 1923 training images
│       ├── valid\images\        ← 205 validation images
│       ├── test\images\         ← test images
│       ├── train\labels\        ← YOLO format annotations
│       ├── valid\labels\
│       └── data.yaml            ← dataset config (paths fixed)
└── hotspot_yolo\
    └── final\
        └── weights\
            ├── best.pt          ← Best model weights ✅
            └── last.pt
```

### ⏭️ Next Steps
- [ ] Run full inference on test set using `best.pt` weights
- [ ] Generate confusion matrix and PR curve visualizations
- [ ] Implement Attention-Driven U-Net segmentation stage (Stage 2)
- [ ] Integrate full cascade pipeline: YOLO detect → crop RoI → U-Net segment
- [ ] Benchmark against baseline models (YOLOv5, YOLOv9, Faster RCNN)

---

## 📚 References
Liu, B., Chen, L., Sun, K., Wang, X., & Zhao, J. (2024). A Hot Spot Identification Approach for Photovoltaic Module Based on Enhanced U-Net With Squeeze-and-Excitation and VGG19. IEEE Transactions on Instrumentation and Measurement.

Ultralytics YOLOv11 (2024). https://docs.ultralytics.com/models/yolo11/

## 📝 License
This project is developed for academic purposes at SASTRA Deemed to be University.
