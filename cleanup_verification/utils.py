
import torch
import numpy as np

def detect_trash(model, image):
    results = model(image)
    result = results[0]

    #  how many objects YOLO detected
    num_detections = len(result.boxes)

    return num_detections
