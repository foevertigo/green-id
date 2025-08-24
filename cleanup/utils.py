import os
from ultralytics import YOLO
from pathlib import Path

# Load model once
model_path = Path(__file__).parent / "model" / "best.pt"
model = YOLO(model_path)

# Class mapping
cls_map = {0: "after", 1: "before"}

def verify_cleanup(before_img_path, after_img_path):
    """Returns True if cleanup verified, False otherwise."""
    res_before = model.predict(before_img_path, verbose=False)[0].probs
    res_after  = model.predict(after_img_path, verbose=False)[0].probs
    
    before_class = cls_map[res_before.top1]
    after_class  = cls_map[res_after.top1]
    
    return before_class == "before" and after_class == "after"

def save_uploaded_file(file, folder):
    """Save uploaded image to folder, return saved path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file.filename)
    file.save(path)
    return path
