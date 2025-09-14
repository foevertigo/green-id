# planting/run_detection.py
"""
Standalone live/demo visual detector (runs in terminal/GUI).
Change LIVE_MODE flag to switch webcam vs demo video.
"""
import cv2, os
from object_detection.detect_objects import Detector, detect_green_blob

LIVE_MODE = False    # True -> webcam, False -> demo video file
DEMO_VIDEO = os.path.join("uploads", "videos", "demo_video.mp4")

def main():
    detector = Detector()  # tries to load YOLO if available
    if LIVE_MODE:
        cap = cv2.VideoCapture(0)
        print("Running LIVE (webcam). Press 'q' to quit.")
    else:
        cap = cv2.VideoCapture(DEMO_VIDEO)
        print(f"Running DEMO ({DEMO_VIDEO}). Press 'q' to quit.")

    if not cap.isOpened():
        print("Cannot open source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or cannot read frame.")
            break

        # detections via YOLO if available
        dets = detector.detect(frame) if detector.model is not None else []
        labels = [d["label"].lower() for d in dets]

        person_present = any("person" in l for l in labels)
        plant_present = any(("plant" in l or "tree" in l or "potted" in l) for l in labels)

        # fallback green blob if YOLO not available or no plant detected
        if not plant_present:
            plant_present = detect_green_blob(frame, roi=None, min_area_px=300)

        status_text = f"Person: {person_present} | Plant: {plant_present} | YOLO_loaded: {detector.model is not None}"
        cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # draw YOLO boxes if any
        if dets:
            for d in dets:
                x1,y1,x2,y2 = d["xyxy"]
                conf = d["conf"]
                label = d["label"]
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("Green-ID Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
