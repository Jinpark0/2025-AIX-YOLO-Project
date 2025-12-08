#!/usr/bin/env python3
"""
Create a smaller BDD100K subset using config.yaml

Usage:
    python make_bdd_subset.py --config config.yaml
"""

import argparse
import json
import random
import shutil
import yaml
from pathlib import Path


# ----------------------------
# Argument Parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config file path"
    )

    return parser.parse_args()


# ----------------------------
# Load YAML Config
# ----------------------------
def load_config(config_path: Path):
    with config_path.open("r") as f:
        return yaml.safe_load(f)


# ----------------------------
# Utility functions
# ----------------------------
def normalize(s):
    return s.lower().strip() if isinstance(s, str) else None


def match_attr(attr_value, wanted):
    """
    wanted:
      - None           -> 무조건 통과
      - "any"          -> 무조건 통과
      - str            -> 해당 값과 정확히 매칭
      - list/tuple[str] -> 리스트 중 하나라도 매칭되면 통과
    """
    if wanted is None:
        return True

    # 리스트인 경우 (예: ["rainy", "snowy"])
    if isinstance(wanted, (list, tuple)):
        norm_attr = normalize(attr_value)
        # 리스트 안에 "any" 가 하나라도 있으면 전부 허용
        if any(isinstance(w, str) and normalize(w) == "any" for w in wanted):
            return True
        return any(norm_attr == normalize(w) for w in wanted)

    # 문자열
    wanted = normalize(wanted)
    if wanted == "any":
        return True
    return normalize(attr_value) == wanted


def find_image_for_label(images_dir: Path, stem: str):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ----------------------------
# Collect filtered candidates
# ----------------------------
def collect_candidates(split, images_root, labels_root, weather, scene, timeofday):
    labels_dir = labels_root / split
    images_dir = images_root / split

    candidates = []

    json_files = sorted(labels_dir.glob("*.json"))
    print(f"[{split}] 전체 label 파일: {len(json_files)}")

    for jpath in json_files:
        with jpath.open("r") as f:
            data = json.load(f)

        attrs = data.get("attributes", {})
        w = attrs.get("weather")
        s = attrs.get("scene")
        t = attrs.get("timeofday")

        if not match_attr(w, weather):
            continue
        if not match_attr(s, scene):
            continue
        if not match_attr(t, timeofday):
            continue

        stem = jpath.stem
        img_path = find_image_for_label(images_dir, stem)
        if img_path:
            candidates.append((stem, jpath, img_path))

    print(f"[{split}] 필터 통과: {len(candidates)}")
    return candidates


# ----------------------------
# Sample & Copy
# ----------------------------
def sample_and_copy(split, candidates, requested_num, out_root):
    out_labels = out_root / "labels" / split
    out_images = out_root / "images" / split
    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    total = len(candidates)
    if total == 0:
        print(f"[{split}] 후보 없음.")
        return 0

    if requested_num <= 0 or requested_num > total:
        num = total
    else:
        num = requested_num

    random.shuffle(candidates)
    selected = candidates[:num]

    print(f"[{split}] {total} 중 {num} 샘플 선택")

    for stem, jpath, img_path in selected:
        shutil.copy2(jpath, out_labels / f"{stem}.json")
        shutil.copy2(img_path, out_images / img_path.name)

    return num


# ----------------------------
# Version Folder 생성
# ----------------------------
def get_next_version_root(base_output_root: Path) -> Path:
    """
    base_output_root 안에 version01, version02 ... 식으로
    이미 있는 폴더를 찾아서, 가장 큰 번호+1 폴더를 새로 만든다.
    """
    base_output_root.mkdir(parents=True, exist_ok=True)

    max_idx = 0
    for p in base_output_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("version"):
            continue
        suffix = name.replace("version", "")
        suffix = suffix.lstrip("0")  # "01" -> "1"
        if suffix.isdigit():
            idx = int(suffix)
            if idx > max_idx:
                max_idx = idx

    new_idx = max_idx + 1
    new_dir = base_output_root / f"version{new_idx:02d}"
    new_dir.mkdir(parents=True, exist_ok=False)

    print(f"=== New output version dir: {new_dir} ===")
    return new_dir


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    if not args.config:
        raise ValueError("--config 파일을 반드시 지정해야 합니다.")

    config = load_config(args.config)
    print("=== Loaded Config ===")
    print(config)

    images_root = Path(config["paths"]["images_root"])
    labels_root = Path(config["paths"]["labels_root"])
    base_output_root = Path(config["paths"]["output_root"])

    seed = config.get("seed", 42)
    random.seed(seed)

    # version01, version02 ... 자동 생성
    output_root = get_next_version_root(base_output_root)

    # ------------------------
    # ① 새 방식: 조건별로 개수 지정
    # ------------------------
    if "conditions" in config:
        conditions = config["conditions"]

        for cond in conditions:
            name = cond.get("name", "cond")
            weather = cond.get("weather")
            scene = cond.get("scene")
            timeofday = cond.get("timeofday")
            samples_cfg = cond.get("samples", {})

            print(f"\n==== Condition: {name} ====")
            print(f"  weather={weather}, scene={scene}, timeofday={timeofday}")

            for split in ["train", "val", "test"]:
                req_num = samples_cfg.get(split, 0)
                if req_num <= 0:
                    continue

                print(f"\n--- {name} / {split} : 요청 {req_num}장 ---")
                candidates = collect_candidates(
                    split,
                    images_root,
                    labels_root,
                    weather,
                    scene,
                    timeofday,
                )
                used = sample_and_copy(split, candidates, req_num, output_root)
                print(f"=== {name} / {split} : 실제 {used}장 사용 ===")

    # ------------------------
    # ② 구 방식: 전체 필터 + 전체 개수
    #    (config에 conditions가 없으면 예전 방식 그대로 동작)
    # ------------------------
    else:
        weather = config["filter"]["weather"]
        scene = config["filter"]["scene"]
        timeofday = config["filter"]["timeofday"]

        samples_cfg = config["samples"]
        num_train = samples_cfg.get("num_train") or samples_cfg.get("train")
        num_val = samples_cfg.get("num_val") or samples_cfg.get("val")
        num_test = samples_cfg.get("num_test") or samples_cfg.get("test")

        splits = {
            "train": num_train,
            "val": num_val,
            "test": num_test,
        }

        for split, req_num in splits.items():
            candidates = collect_candidates(
                split,
                images_root,
                labels_root,
                weather,
                scene,
                timeofday,
            )
            sample_and_copy(split, candidates, req_num, output_root)


if __name__ == "__main__":
    main()
