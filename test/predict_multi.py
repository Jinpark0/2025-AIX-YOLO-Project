from ultralytics import YOLO
from pathlib import Path
import cv2
import torch
import numpy as np
# import albumentations as A


model = YOLO("./runs/detect/train11/weights/best.pt")
# model = YOLO("yolo11n.pt")

sub_yaml_files = [
    "day_clear.yaml",
    "day_rain.yaml",
    "day_snow.yaml",
    "night_clear.yaml",
    "night_rain.yaml",
    "night_snow.yaml",
]

results_summary = []

for yaml_path in sub_yaml_files:
    subset_name = Path(yaml_path).stem  # day_clear 이런 이름

    print(f"\n==================== {subset_name} ====================")
    metrics = model.val(
        data=yaml_path,
        split="test",
        imgsz=640,
        device=0,
        verbose=False, 
    )


    p = float(metrics.box.mp)      
    r = float(metrics.box.mr)        
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    print(f"{subset_name} -> P={p:.3f}, R={r:.3f}, mAP50={map50:.3f}, mAP50-95={map5095:.3f}")

    results_summary.append((subset_name, p, r, map50, map5095))

# 마지막에 한 번에 표로 요약 출력
print("\n=== Per-subset evaluation summary ===")
print(f"{'Subset':<12} {'P':>6} {'R':>6} {'mAP50':>8} {'mAP50-95':>10}")
print("-" * 46)
for name, p, r, map50, map5095 in results_summary:
    print(f"{name:<12} {p:>6.3f} {r:>6.3f} {map50:>8.3f} {map5095:>10.3f}")
