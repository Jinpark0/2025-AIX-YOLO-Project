#!/usr/bin/env python3
import os
import random
from pathlib import Path

import cv2
import numpy as np


# ========= 설정 =========
DATASET_ROOT = Path("./dataset_yolo_converted")  # 너의 dataset 루트
SPLIT = "train"          # "train", "val", "test" 중 하나
NUM_SAMPLES = 10         # 시각화할 이미지 개수
OUTPUT_DIR = Path("./yolo_viz")  # 결과 이미지 저장 폴더
# =======================


def find_image_files(images_dir: Path):
    exts = [".jpg", ".jpeg", ".png"]
    return [p for p in images_dir.iterdir() if p.suffix.lower() in exts]


def load_yolo_labels(label_path: Path):
    """
    YOLO txt (class x_center y_center width height, 모두 0~1) 파싱
    반환: 리스트 [ (cls, x_c, y_c, w, h), ... ]
    """
    boxes = []
    if not label_path.exists():
        return boxes

    with label_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            boxes.append((cls, x_c, y_c, w, h))
    return boxes


def draw_boxes(img, boxes):
    """
    img: OpenCV 이미지 (H, W, 3)
    boxes: YOLO normalized (cls, x_c, y_c, w, h)
    """
    h, w = img.shape[:2]

    for cls, x_c, y_c, bw, bh in boxes:
        # YOLO normalized -> 픽셀 좌표
        cx = x_c * w
        cy = y_c * h
        bw_px = bw * w
        bh_px = bh * h

        x1 = int(cx - bw_px / 2)
        y1 = int(cy - bh_px / 2)
        x2 = int(cx + bw_px / 2)
        y2 = int(cy + bh_px / 2)

        # 이미지 범위 안으로 클램프
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        # 박스 그리기 (초록 테두리, 두께 2px)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # class id 텍스트 표시
        label = str(cls)
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )

    return img


def main():
    images_dir = DATASET_ROOT / "images" / SPLIT
    labels_dir = DATASET_ROOT / "labels" / SPLIT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir():
        raise RuntimeError(f"이미지 폴더 없음: {images_dir}")
    if not labels_dir.is_dir():
        raise RuntimeError(f"라벨 폴더 없음: {labels_dir}")

    image_files = find_image_files(images_dir)

    # label이 있는 파일만 사용
    valid_pairs = []
    for img_path in image_files:
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))

    if not valid_pairs:
        raise RuntimeError("라벨이 있는 이미지가 한 개도 없습니다… 경로 확인해줘!")

    random.seed(123)  # 고정하면 매번 같은 10장
    random.shuffle(valid_pairs)

    selected = valid_pairs[: min(NUM_SAMPLES, len(valid_pairs))]
    print(f"{SPLIT} split에서 {len(selected)}장 시각화합니다.")

    for img_path, label_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"이미지 읽기 실패: {img_path}")
            continue

        boxes = load_yolo_labels(label_path)
        img_drawn = draw_boxes(img, boxes)

        out_name = img_path.stem + "_viz.jpg"
        out_path = OUTPUT_DIR / out_name
        cv2.imwrite(str(out_path), img_drawn)

        print(f"  - {img_path.name} -> {out_path}")

    print(f"\n완료! '{OUTPUT_DIR}' 폴더에서 박스가 잘 그려졌는지 확인해봐.")


if __name__ == "__main__":
    main()
