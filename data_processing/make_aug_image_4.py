import os
import glob
import random
import shutil
import cv2
import albumentations as A
from tqdm import tqdm

# ==============================
# 1. 경로 설정
# ==============================

input_image_dir = "bdd_to_yolo/dataset_yolo_converted/images/train"
input_label_dir = "bdd_to_yolo/dataset_yolo_converted/labels/train"
output_root_dir = "./bdd100k_augmented"

output_image_dir = os.path.join(output_root_dir, "images")
output_label_dir = os.path.join(output_root_dir, "labels")

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# ==============================
# 2. 옵션
# ==============================

# "rain", "snow", 또는 None (복사만)
aug_type = "snow"   

n_aug_per_image = 2.3     # aug_type이 None일 때는 무시됨

# ==============================
# 3. Transform 정의
# ==============================

rain_transform = A.Compose([
    A.RandomRain(
        slant_range=[-15, 15],
        drop_length=60,
        drop_width=1,
        drop_color=[200, 200, 200],
        blur_value=7,
        brightness_coefficient=0.5,
        rain_type="heavy",
        p=1.0
    )
])

snow_transform = A.Compose([
    A.RandomSnow(
        brightness_coeff=4,
        snow_point_range=[0.3, 0.7],
        method="bleach",
        p=1.0
    )
])

def get_transform(aug_type):
    if aug_type == "rain":
        return rain_transform
    elif aug_type == "snow":
        return snow_transform
    elif aug_type is None:
        return None
    else:
        raise ValueError("aug_type must be 'rain', 'snow', or None")

transform = get_transform(aug_type)

# ==============================
# 4. 이미지 리스트 가져오기
# ==============================

image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    image_paths.extend(glob.glob(os.path.join(input_image_dir, ext)))
image_paths = sorted(image_paths)

if len(image_paths) == 0:
    raise RuntimeError("이미지가 없습니다.")

print(f"총 원본 이미지: {len(image_paths)}")


# ==============================
# 5. Loop 시작
# ==============================
if aug_type is None:
    print("### AUGMENTATION OFF MODE ###")
    print("→ 이미지를 그대로 복사만 합니다.")
else:
    print(f"### AUGMENTATION MODE: {aug_type} ###")

base_n = int(n_aug_per_image)
extra_prob = n_aug_per_image - base_n

for img_path in tqdm(image_paths, desc="Processing"):

    img_name = os.path.basename(img_path)
    base_name, ext = os.path.splitext(img_name)

    label_path = os.path.join(input_label_dir, base_name + ".txt")

    if not os.path.exists(label_path):
        print(f"[경고] 라벨 없음: {label_path}")
        continue

    # ==========================================
    # 5-1) 원본 이미지 복사 (누적 실행 대비)
    # ==========================================
    out_img_path = os.path.join(output_image_dir, img_name)
    out_lbl_path = os.path.join(output_label_dir, base_name + ".txt")

    if not os.path.exists(out_img_path):
        shutil.copy2(img_path, out_img_path)
    if not os.path.exists(out_lbl_path):
        shutil.copy2(label_path, out_lbl_path)

    # ==========================================
    # 5-2) 증강 안 한다면 여기서 종료
    # ==========================================
    if aug_type is None:
        continue

    # ==========================================
    # 5-3) 증강 개수 결정
    # ==========================================
    n_aug = base_n
    if random.random() < extra_prob:
        n_aug += 1

    image = cv2.imread(img_path)

    # 이미 존재하는 aug 개수 카운트 (누적 실행 안전)
    existing_aug = glob.glob(
        os.path.join(output_image_dir, f"{base_name}_aug-{aug_type}-*{ext}")
    )
    start_idx = len(existing_aug)

    # ==========================================
    # 5-4) augmentation 생성
    # ==========================================
    for i in range(n_aug):
        aug = transform(image=image)
        aug_img = aug["image"]

        aug_index = start_idx + i + 1
        aug_base_name = f"{base_name}_aug-{aug_type}-{aug_index}"

        aug_img_path = os.path.join(output_image_dir, aug_base_name + ext)
        aug_lbl_path = os.path.join(output_label_dir, aug_base_name + ".txt")

        cv2.imwrite(aug_img_path, aug_img)
        shutil.copy2(label_path, aug_lbl_path)

print("완료!")

