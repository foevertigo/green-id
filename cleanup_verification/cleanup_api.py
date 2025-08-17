

from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import os
from utils import detect_trash
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel

# Safelist YOLO's DetectionModel
add_safe_globals([DetectionModel])
app = Flask(__name__)


model = YOLO("taco_best.pt")  

@app.route("/cleanup-check", methods=["POST"])
def cleanup_check():
    if "before" not in request.files or "after" not in request.files:
        return jsonify({"error": "Please upload both 'before' and 'after' images."}), 400

    before = Image.open(request.files["before"].stream)
    after = Image.open(request.files["after"].stream)

    before_count = detect_trash(model, before)
    after_count = detect_trash(model, after)

   
    if after_count < before_count:
        return jsonify({
            "status": "success",
            "before_trash": before_count,
            "after_trash": after_count,
            "message": "Cleanup detected",
            "points_awarded": 50
        })
    else:
        return jsonify({
            "status": "fail",
            "before_trash": before_count,
            "after_trash": after_count,
            "message": "No cleanup detected",
            "points_awarded": 0
        })

if __name__ == "__main__":
    app.run(debug=True)
