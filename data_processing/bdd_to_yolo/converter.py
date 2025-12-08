import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# config 파일에서 필요한 변수들을 가져옵니다.
# IMAGES_ROOT, LABELS_ROOT, OUTPUT_DATASET_DIR, USE_ULTRALYTICS_FORMAT가 정의되어 있어야 합니다.
from config import IMAGES_ROOT, LABELS_ROOT, OUTPUT_DATASET_DIR, USE_ULTRALYTICS_FORMAT

# ============================================================
# 설정 및 상수
# ============================================================

# 객체 탐지에 사용하지 않을 카테고리
IGNORED_CATEGORIES = {
    'lane',
    'drivable area',
    'area/drivable',
    'area/alternative'
}

# BDD100K 이미지 크기
IMG_SIZE = (1280, 720)

# 🔒 고정 클래스 매핑 (19 classes)
# YAML, 모델(nc=19), 라벨 모두 이 기준을 따라야 함
FIXED_CATEGORIES = {
    'area/unknown': 0,
    'bike': 1,
    'bus': 2,
    'car': 3,
    'lane/crosswalk': 4,
    'lane/double other': 5,
    'lane/double white': 6,
    'lane/double yellow': 7,
    'lane/road curb': 8,
    'lane/single other': 9,
    'lane/single white': 10,
    'lane/single yellow': 11,
    'motor': 12,
    'person': 13,
    'rider': 14,
    'traffic light': 15,
    'traffic sign': 16,
    'train': 17,
    'truck': 18,
}

# ============================================================
# 라벨 변환
# ============================================================

def convert_split_labels(json_input_dir, label_output_dir, categories, img_size):
    """JSON → YOLO txt 변환"""
    img_w, img_h = img_size
    label_output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(Path(json_input_dir).glob("*.json"))

    for json_path in tqdm(json_files, desc=f"Converting {json_input_dir.name} labels"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        label_path = label_output_dir / f"{json_path.stem}.txt"

        if (
            "frames" not in data
            or not data["frames"]
            or "objects" not in data["frames"][0]
        ):
            continue

        objects = data["frames"][0]["objects"]

        with open(label_path, "w") as out:
            for obj in objects:
                cat = obj.get("category")

                if (
                    cat in IGNORED_CATEGORIES
                    or cat not in categories
                    or "box2d" not in obj
                ):
                    continue

                box = obj["box2d"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

                # YOLO normalized bbox
                x_c = (x1 + x2) / 2 / img_w
                y_c = (y1 + y2) / 2 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                # clip
                x_c = max(0.0, min(1.0, x_c))
                y_c = max(0.0, min(1.0, y_c))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                out.write(
                    f"{categories[cat]} "
                    f"{x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n"
                )

# ============================================================
# 유틸
# ============================================================

def copy_images(src_img_dir, dst_img_dir):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    for img in Path(src_img_dir).glob("*.jpg"):
        shutil.copy(img, dst_img_dir / img.name)

def generate_split_txt(img_dir, label_dir, output_txt):
    with open(output_txt, "w") as f:
        for img in sorted(Path(img_dir).glob("*.jpg")):
            label = label_dir / f"{img.stem}.txt"
            if label.exists() and os.path.getsize(label) > 0:
                f.write(str(img.resolve()) + "\n")

# ============================================================
# 설정 파일 생성
# ============================================================

def generate_yaml(output_root, yaml_path, categories, use_ultralytics=True):
    if use_ultralytics:
        train = "images/train"
        val = "images/val"
        test = "images/test"
    else:
        train = val = test = ""

    lines = [
        f"path: {output_root}",
        f"train: {train}",
        f"val: {val}",
        f"test: {test}",
        "names:"
    ]

    for name, idx in sorted(categories.items(), key=lambda x: x[1]):
        lines.append(f"  {idx}: {name}")

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

def generate_darknet_files(out_dir, split_txts, categories):
    names_path = out_dir / "bdd100k.names"
    data_path = out_dir / "bdd100k.data"

    with open(names_path, "w") as f:
        for name, _ in sorted(categories.items(), key=lambda x: x[1]):
            f.write(name + "\n")

    with open(data_path, "w") as f:
        f.write(f"classes = {len(categories)}\n")
        f.write(f"train = {split_txts['train']}\n")
        f.write(f"valid = {split_txts['val']}\n")
        f.write(f"test = {split_txts['test']}\n")
        f.write(f"names = {names_path.resolve()}\n")

# ============================================================
# Main
# ============================================================

def main():
    categories = FIXED_CATEGORIES

    print("\n✅ Using FIXED_CATEGORIES:")
    for name, idx in sorted(categories.items(), key=lambda x: x[1]):
        print(f"  {idx:2d}: {name}")

    yolo_files_dir = OUTPUT_DATASET_DIR / "yolo_files"
    yolo_files_dir.mkdir(parents=True, exist_ok=True)
    split_txts = {}

    for split in ["train", "val", "test"]:
        src_imgs = IMAGES_ROOT / split
        json_src = LABELS_ROOT / split

        if USE_ULTRALYTICS_FORMAT:
            img_dst = OUTPUT_DATASET_DIR / "images" / split
            lbl_dst = OUTPUT_DATASET_DIR / "labels" / split
        else:
            img_dst = OUTPUT_DATASET_DIR / split / "images"
            lbl_dst = OUTPUT_DATASET_DIR / split / "labels"

        print(f"\n🛠️ {split} 변환 중...")
        copy_images(src_imgs, img_dst)
        convert_split_labels(json_src, lbl_dst, categories, IMG_SIZE)

        split_txt = yolo_files_dir / f"{split}.txt"
        generate_split_txt(img_dst, lbl_dst, split_txt)
        split_txts[split] = split_txt.resolve()

    generate_darknet_files(yolo_files_dir, split_txts, categories)

    yaml_path = OUTPUT_DATASET_DIR / "bdd100k_ultralytics.yaml"
    generate_yaml(
        OUTPUT_DATASET_DIR.resolve(),
        yaml_path,
        categories,
        USE_ULTRALYTICS_FORMAT
    )

    print("\n=== DONE ===")
    print(f"✅ YAML: {yaml_path}")
    print(f"✅ YOLO files: {yolo_files_dir}")
    print("이제 이 데이터셋으로만 학습/평가하세요.")

if __name__ == "__main__":
    main()
