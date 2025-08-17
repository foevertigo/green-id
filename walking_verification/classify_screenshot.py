import cv2
import joblib
import pytesseract
import os

# Setting the path to the Tesseract executable
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if not os.path.exists(tesseract_path):
    raise FileNotFoundError(f"Tesseract not found at {tesseract_path}")
pytesseract.pytesseract.tesseract_cmd = tesseract_path
import numpy as np
from PIL import Image
import re

# Loading the trained model once
model = joblib.load(r"C:\Projects\Green-ID\walking_verification\fit_model.pkl")


def classify_walking_screenshot(image_path):
    # Load and preprocess image
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (10, 10)).flatten().reshape(1, -1)
    
    # Predict
    prediction = model.predict(resized)[0]
    
    # Run OCR
    text = pytesseract.image_to_string(Image.open(image_path)).lower()
    
    # Step keyword + number check
    has_steps = "steps" in text
    has_number = bool(re.search(r"\d{3,}", text))  # 3+ digit number
    
    is_valid = prediction == 1 and has_steps and has_number

    return {
        "prediction": "Google Fit / Apple Screenshot" if prediction else "Not recognized",
        "steps_keyword": has_steps,
        "step_number_found": has_number,
        "final_verdict": "Valid Screenshot" if is_valid else "Invalid Screenshot"
    }
