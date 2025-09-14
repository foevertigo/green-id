# planting/video_processing/verify_video.py
import os, cv2, numpy as np
from object_detection.detect_objects import Detector, detect_green_blob
from utils.cleanup import ensure_empty_dir

def mean_optical_flow(prev_gray, gray, roi=None):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    if roi:
        x1,y1,x2,y2 = roi
        h, w = mag.shape
        x1,y1 = max(0,int(x1)), max(0,int(y1))
        x2,y2 = min(w,int(x2)), min(h,int(y2))
        if x2 <= x1 or y2 <= y1:
            crop = mag
        else:
            crop = mag[y1:y2, x1:x2]
    else:
        crop = mag
    return float(np.mean(crop))

def verify_planting_from_frames(frames_folder, min_plant_frames=1, motion_threshold=0.8):
    detector = Detector()
    frame_files = sorted([os.path.join(frames_folder,f) for f in os.listdir(frames_folder) if f.lower().endswith(".jpg")])
    if not frame_files:
        return False, {"reason": "no_frames"}

    prev_gray = None
    person_frames = []
    plant_frames = []
    motion_scores = []

    for idx, fp in enumerate(frame_files):
        img = cv2.imread(fp)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dets = detector.detect(img) if detector.model is not None else []
        persons = [d for d in dets if 'person' in d['label'].lower()]
        plants = [d for d in dets if ('plant' in d['label'].lower() or 'potted' in d['label'].lower() or 'tree' in d['label'].lower())]

        person_bbox = None
        if persons:
            person = max(persons, key=lambda d: (d['xyxy'][2]-d['xyxy'][0])*(d['xyxy'][3]-d['xyxy'][1]))
            person_frames.append(idx)
            person_bbox = person['xyxy']

        if plants:
            plant_frames.append(idx)
        else:
            # fallback: green blob near person's lower area
            if person_bbox is not None:
                x1,y1,x2,y2 = person_bbox
                h = img.shape[0]
                roi = (max(0,x1-20), min(h-1,y2), min(img.shape[1], x2+20), min(h, y2 + (y2-y1)//2 + 30))
                if detect_green_blob(img, roi=roi, min_area_px=150):
                    plant_frames.append(idx)
            else:
                if detect_green_blob(img, roi=None, min_area_px=800):
                    plant_frames.append(idx)

        if prev_gray is not None:
            # measure motion around person if exists else whole frame
            if person_bbox is not None:
                x1,y1,x2,y2 = person_bbox
                ry1 = y1 + (y2-y1)//2
                roi = (x1, ry1, x2, y2)
                motion = mean_optical_flow(prev_gray, gray, roi=roi)
            else:
                motion = mean_optical_flow(prev_gray, gray, roi=None)
            motion_scores.append(motion)
        prev_gray = gray

    evidence = {
        "num_frames": len(frame_files),
        "person_frames": person_frames,
        "plant_frames": plant_frames,
        "avg_motion": float(sum(motion_scores)/len(motion_scores)) if motion_scores else 0.0,
        "motion_samples": motion_scores[:10]
    }

    if not person_frames:
        return False, {"reason":"no_person_detected", **evidence}
    if len(plant_frames) < min_plant_frames:
        return False, {"reason":"no_plant_detected", **evidence}
    if evidence["avg_motion"] < motion_threshold:
        return False, {"reason":"insufficient_ground_motion", **evidence}

    return True, {"reason":"planting_verified", **evidence}

if __name__ == "__main__":
    # quick local run: extract frames then verify
    from video_processing.extract_frames import extract_frames
    ensure_empty_dir("uploads/frames")
    frames = extract_frames("uploads/videos/demo_video.mp4", out_dir="uploads/frames", sample_fps=1, max_frames=30)
    ok, info = verify_planting_from_frames("uploads/frames", min_plant_frames=1, motion_threshold=0.6)
    print("PASS:", ok)
    print(info)
