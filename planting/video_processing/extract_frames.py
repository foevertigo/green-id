import cv2
import os

def extract_frames(video_path, output_folder, interval=10):
    print(f"xtracting frames from: {video_path}")
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            cv2.imwrite(f"{output_folder}/frame_{frame_id}.jpg", frame)
            frame_id += 1
        count += 1

    cap.release()
    print(f"Extracted {frame_id} frames to: {output_folder}")

