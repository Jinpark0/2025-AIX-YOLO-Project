from pathlib import Path

# === CONFIGURATION FILE FOR BDD100K TO YOLO FORMAT CONVERSION ===

# Path to the BDD100K images directory containing train/val/test folders with .jpg images
IMAGES_ROOT = Path("./bdd100k_subset_filtered/version01/images")

# Path to the BDD100K labels directory containing the JSON files (train/val/test.json)
LABELS_ROOT = Path("./bdd100k_subset_filtered/version01/labels")

# Destination folder where YOLO-formatted dataset will be saved
OUTPUT_DATASET_DIR = Path("./dataset_yolo_converted")

# Flag: Set to True to generate dataset in Ultralytics YOLO format (YOLOv7+)
# Set to False to generate legacy YOLO format
USE_ULTRALYTICS_FORMAT = True
