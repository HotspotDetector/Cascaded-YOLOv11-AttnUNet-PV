# HOTSPOT-YOLO Offline Inference
# Run this completely offline after downloading best_200.pt from Google Drive
# No internet required after setup

# ============================================================
# CELL 1: Install (only needed once)
# ============================================================
# !pip install ultralytics opencv-python matplotlib --quiet

# ============================================================
# CELL 2: Imports
# ============================================================
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import glob

print("✅ Imports done!")

# ============================================================
# CELL 3: Load Your Downloaded Model
# ============================================================

# 👇 Change this to wherever you saved best_200.pt
MODEL_PATH = r"C:\Users\visho\Downloads\Demo\best_200.pt"

# 👇 Change this to your test images folder
TEST_IMAGES_PATH = r"C:\Users\visho\Downloads\Demo\Dataset\test\images"

model = YOLO(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")

test_images = (
    glob.glob(os.path.join(TEST_IMAGES_PATH, "*.jpg")) +
    glob.glob(os.path.join(TEST_IMAGES_PATH, "*.png"))
)
print(f"✅ Found {len(test_images)} test images")

# ============================================================
# CELL 4: Run Inference
# ============================================================

results = model.predict(
    source = TEST_IMAGES_PATH,
    conf   = 0.25,
    iou    = 0.45,
    save   = True,
    project= "hotspot_predictions",
    name   = "offline_run"
)

print(f"✅ Predictions saved to: hotspot_predictions/offline_run/")

# ============================================================
# CELL 5: Visualize Detections (6 images)
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, result in enumerate(results[:6]):
    annotated     = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    n             = len(result.boxes)
    confs         = result.boxes.conf.cpu().numpy() if n > 0 else []
    avg_conf      = np.mean(confs) * 100 if len(confs) > 0 else 0

    axes[idx].imshow(annotated_rgb)
    axes[idx].set_title(
        f"Image {idx+1} | Hotspots: {n} | Confidence: {avg_conf:.1f}%",
        fontsize=10, fontweight='bold'
    )
    axes[idx].axis('off')

for i in range(len(results[:6]), 6):
    axes[i].axis('off')

plt.suptitle(
    'HOTSPOT-YOLO: Thermal Anomaly Detections (200 epochs | 87% mAP)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('offline_detections.png', dpi=150)
plt.show()
print("✅ Saved: offline_detections.png")

# ============================================================
# CELL 6: Final Metrics Bar Chart
# ============================================================

metrics = {
    'mAP@50 (%)':    87.0,
    'mAP@50-95 (%)': 58.8,
    'Precision (%)': 86.7,
    'Recall (%)':    82.1
}

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ecc71', '#1abc9c', '#9b59b6', '#f39c12']
bars = ax.bar(metrics.keys(), metrics.values(), color=colors, width=0.5)

ax.set_title('HOTSPOT-YOLO Final Results (200 epochs)', fontsize=14, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_ylim(0, 100)

for bar, val in zip(bars, metrics.values()):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f'{val}%', ha='center', fontweight='bold', fontsize=12
    )

# Add base paper reference line
ax.axhline(y=90.8, color='red', linestyle='--', linewidth=2, label='Base Paper (90.8%)')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('final_metrics.png', dpi=150)
plt.show()
print("✅ Saved: final_metrics.png")

# ============================================================
# CELL 7: Evaluate on Test Set (official metrics)
# ============================================================

# Point to your local data.yaml
DATA_YAML = r"C:\Users\visho\Downloads\Demo\Dataset\data.yaml"

eval_results = model.val(
    data  = DATA_YAML,
    split = 'test',
    conf  = 0.25,
    iou   = 0.45,
    plots = True
)

print("\n" + "="*50)
print("     HOTSPOT-YOLO OFFICIAL TEST RESULTS")
print("="*50)
print(f"  mAP@50      : {eval_results.results_dict['metrics/mAP50(B)']*100:.1f}%")
print(f"  mAP@50-95   : {eval_results.results_dict['metrics/mAP50-95(B)']*100:.1f}%")
print(f"  Precision   : {eval_results.results_dict['metrics/precision(B)']*100:.1f}%")
print(f"  Recall      : {eval_results.results_dict['metrics/recall(B)']*100:.1f}%")
print("="*50)
print("\n  Base Paper  : 90.8%")
print(f"  Our Model   : {eval_results.results_dict['metrics/mAP50(B)']*100:.1f}%")
print(f"  Gap         : {90.8 - eval_results.results_dict['metrics/mAP50(B)']*100:.1f}%")
print("="*50)
