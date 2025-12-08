#!/usr/bin/env python3
"""
Train YOLO11 with Albumentations weather augmentations
(RandomSnow + RandomRain)
"""

import albumentations as A
from ultralytics import YOLO


# -----------------------------------------
# Weather augmentations (your coefficients)
# -----------------------------------------
def get_weather_augmentations():
    return [
        # 🌨️ Snow
        A.RandomSnow(
            brightness_coeff=4,
            snow_point_range=(0.3, 0.7),  # 반드시 tuple
            method="bleach",
            p=0.5,
        ),

        # 🌧️ Rain
        A.RandomRain(
            slant_range=(-15, 15),        # list/tuple 모두 OK
            drop_length=60,               # ❗ int
            drop_width=1,
            drop_color=(200, 200, 200),   # tuple 권장
            blur_value=7,
            brightness_coefficient=0.5,
            rain_type="heavy",
            p=0.5,
        ),
    ]


def main():
    yaml_path = "../bdd_to_yolo/dataset_yolo_converted/bdd100k_ultralytics.yaml"

    model = YOLO("yolo11n.pt")

    weather_augs = get_weather_augmentations()

    model.train(
        data=yaml_path,
        epochs=40,
        imgsz=640,
        batch=16,
        device=0,
        project="runs/detect",
        name="weather_augmented_v1",
        save=True,
        plots=True,
        augmentations=weather_augs,   # ✅ 계수 적용
    )

    metrics = model.val()
    print(f"Validation mAP50: {metrics.box.map50:.4f}")

    path = model.export(format="onnx")
    print(f"Model exported: {path}")


if __name__ == "__main__":
    main()
