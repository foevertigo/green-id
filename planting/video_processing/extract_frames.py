# planting/video_processing/extract_frames.py
import cv2, os
from pathlib import Path

def extract_frames(video_path, out_dir="uploads/frames", sample_fps=1, max_frames=None):
    """
    Extract frames from video_path into out_dir.
    sample_fps: frames per second to save (1 => save 1 fps)
    max_frames: maximum total frames to extract (None => no limit)
    Returns list of saved frame paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(video_fps / float(sample_fps))))
    frame_paths = []
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            fname = Path(out_dir) / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(fname), frame)
            frame_paths.append(str(fname))
            saved += 1
            if max_frames and saved >= max_frames:
                break
        idx += 1
    cap.release()
    return frame_paths

if __name__ == "__main__":
    frames = extract_frames("uploads/videos/demo_video.mp4", out_dir="uploads/frames", sample_fps=1, max_frames=30)
    print("Extracted frames:", len(frames))
