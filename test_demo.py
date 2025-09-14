# test_demo.py
import os
from planting.video_processing.extract_frames import extract_frames
from planting.object_detection.detect_objects import Detector, detect_green_blob
from planting.utils.check_gps import validate_gps




# --- CONFIG ---
VIDEO_PATH = "planting/uploads/videos/demo_video.mp4"
FRAMES_DIR = "planting/uploads/frames/demo_video"
SAMPLE_FPS = 2          # extract every 2nd frame (adjust for speed)
MIN_PLANT_FRAMES = 5    # minimum frames with plant detection to pass
ALLOWED_BBOX = (28.40, 28.90, 76.80, 77.50)  # mock GPS bounding box
DEMO_GPS = {"lat": 28.6139, "lon": 77.2090}  # mock GPS for demo/testing

# --- CREATE FRAMES DIR ---
os.makedirs(FRAMES_DIR, exist_ok=True)

# --- EXTRACT FRAMES ---
print(f"[FRAME EXTRACTION] Extracting frames from {VIDEO_PATH}...")
frame_paths = extract_frames(VIDEO_PATH, FRAMES_DIR, sample_fps=SAMPLE_FPS)
print(f"[FRAME EXTRACTION] {len(frame_paths)} frames extracted.")

# --- INIT DETECTOR ---
detector = Detector(model_path="yolov8n.pt", conf=0.25)

# --- DETECTION LOOP ---
plant_frames = 0
for frame_path in frame_paths:
    import cv2
    frame = cv2.imread(frame_path)
    detections = detector.detect(frame)

    # simple fallback green blob detection for demo
    plant_detected = detect_green_blob(frame)
    if plant_detected:
        plant_frames += 1

print(f"[DETECTION] Plant detected in {plant_frames} frames out of {len(frame_paths)}")

# --- MOTION CHECK ---
motion_ok = plant_frames >= MIN_PLANT_FRAMES
print(f"[MOTION] Planting motion verified: {motion_ok}")

# --- GPS CHECK ---
gps_ok, gps_info = validate_gps(gps_json=DEMO_GPS, allowed_bbox=ALLOWED_BBOX)
print(f"[GPS CHECK] {gps_ok}, info={gps_info}")

# --- FINAL VERIFICATION ---
if motion_ok and gps_ok:
    print("✅ Planting verification passed!")
else:
    print("❌ Verification failed!")
