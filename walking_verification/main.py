import cv2
import joblib
import pytesseract
import os
import numpy as np

from PIL import Image
import re

# ---- Tesseract Setup ----
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if not os.path.exists(tesseract_path):
    raise FileNotFoundError(f"Tesseract not found at {tesseract_path}")
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# ---- Load Model ----
model_path = r"C:\Projects\Green-ID\walking_verification\fit_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")
model = joblib.load(model_path)

# ---- Image Path ----
img_path = r"C:\Projects\Green-ID\walking_verification\test_images\IMG-20250531-WA0200.jpg"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found at {img_path}")

# ---- Image Preprocessing ----
image = cv2.imread(img_path)
resized = cv2.resize(image, (224, 224)).flatten().reshape(1, -1)

# ---- Prediction ----
prediction = model.predict(resized)[0]
is_fitness_ss = prediction == 1

# ---- OCR ----
text = pytesseract.image_to_string(Image.open(img_path)).lower()

# ---- Step Count Checks ----
has_steps = "steps" in text
has_number = bool(re.search(r"\d{3,}", text))
is_valid = is_fitness_ss and has_steps and has_number

# ---- Final Output ----
print("\n--- WALKING SCREENSHOT VERIFICATION ---")
print(f"Model Prediction      : {'Google Fit / Apple Screenshot' if is_fitness_ss else 'Not recognized'}")
print(f"Steps Keyword Found   : {has_steps}")
print(f"Step Count Number     : {has_number}")
print(f"Final Verdict         : {'Valid Screenshot' if is_valid else 'Invalid Screenshot'}")
