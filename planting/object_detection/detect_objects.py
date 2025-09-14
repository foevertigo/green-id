# planting/object_detection/detect_objects.py
import cv2
import numpy as np
import os

# Try to import ultralytics YOLO; if not available use fallback
ULTRALYTICS_AVAILABLE = False
YOLO_LOAD_OK = False
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    # Workaround for PyTorch safe unpickling issues if torch is present
    try:
        import torch
        from ultralytics.nn.tasks import DetectionModel
        # allowlist the DetectionModel for torch.load if needed
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([DetectionModel])
    except Exception:
        pass
except Exception:
    ULTRALYTICS_AVAILABLE = False

class Detector:
    """
    Wrapper: tries YOLO (ultralytics) if available, otherwise model=None.
    Methods:
      - detect(frame): returns list of detections [{'label','conf','xyxy'}]
    """
    def __init__(self, model_path="yolov8n.pt", device="cpu", conf=0.25):
        self.conf = conf
        self.device = device
        self.model = None
        if ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                global YOLO_LOAD_OK
                YOLO_LOAD_OK = True
            except Exception:
                self.model = None
                YOLO_LOAD_OK = False

    def detect(self, frame):
        detections = []
        if self.model is None:
            return detections
        try:
            results = self.model(frame, imgsz=640, conf=self.conf, verbose=False)
            res = results[0]
            # ultralytics returns Boxes object on res.boxes
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                return detections
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            names = getattr(self.model, "names", {})
            for (b, c, cid) in zip(xyxy, confs, cls_ids):
                x1, y1, x2, y2 = map(int, b.tolist())
                label = names.get(cid, str(cid))
                detections.append({"label": label, "conf": float(c), "xyxy": (x1, y1, x2, y2)})
        except Exception:
            # If ultralytics API differences or errors occur, return empty list
            return []
        return detections

def detect_green_blob(frame, roi=None, min_area_px=300):
    """
    Fast fallback: detect large green regions (sapling/leaves).
    roi: (x1,y1,x2,y2) in pixel coords
    """
    if frame is None:
        return False
    h, w = frame.shape[:2]
    if roi:
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            crop = frame
        else:
            crop = frame[y1:y2, x1:x2]
    else:
        crop = frame

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    low = np.array([25, 35, 35])
    high = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, low, high)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area_px:
            return True
    return False

if __name__ == "__main__":
    # quick local smoke test
    print("Ultralytics available:", ULTRALYTICS_AVAILABLE, "YOLO loaded:", YOLO_LOAD_OK)
    import numpy as np
    img = np.zeros((240,320,3), np.uint8)
    cv2.rectangle(img, (50,50), (270,190), (0,200,0), -1)
    print("Green blob fallback:", detect_green_blob(img))
