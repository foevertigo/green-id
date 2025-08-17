import json
import os
from PIL import Image


ANNOTATIONS_PATH = "TACO-master/data/annotations.json"
IMAGES_DIR = "cleanup/taco_data/images"
YOLO_ANN_DIR = "cleanup/taco_data/annotations"

os.makedirs(YOLO_ANN_DIR, exist_ok=True)

# === Mapping Category ID to YOLO Class ID ===
def get_category_mapping(annotations_json):
    categories = annotations_json["categories"]
    print(f"Found categories: {[cat['name'] for cat in categories]}")
    return {cat["id"]: idx for idx, cat in enumerate(categories)}

# === Loading Annotations ===
with open(ANNOTATIONS_PATH, "r") as f:
    data = json.load(f)

category_map = get_category_mapping(data)
image_id_map = {img["id"]: img for img in data["images"]}

created_labels = 0
skipped_annotations = 0
missing_images = []

# ===COCO bbox to YOLO format ===
for ann in data["annotations"]:
    img_id = ann["image_id"]
    img_info = image_id_map.get(img_id)

    if not img_info:
        skipped_annotations += 1
        continue

    img_name = img_info["file_name"].split("/")[-1].lower()
    img_path = os.path.join(IMAGES_DIR, img_name)

    if not os.path.exists(img_path):
        missing_images.append(img_name)
        skipped_annotations += 1
        continue

    try:
        w, h = Image.open(img_path).size
    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
        skipped_annotations += 1
        continue

    # bbox = [x_min, y_min, width, height]
    x_min, y_min, width, height = ann["bbox"]
    x_center = (x_min + width / 2) / w
    y_center = (y_min + height / 2) / h
    norm_width = width / w
    norm_height = height / h

    class_id = category_map[ann["category_id"]]
    label_file = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(YOLO_ANN_DIR, label_file)

    with open(label_path, "a") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

    created_labels += 1


print("\nAll done!")
print(f"YOLO labels created: {created_labels}")
print(f"Annotations skipped: {skipped_annotations}")
if missing_images:
    print(f"Missing image samples: {missing_images[:10]}")
