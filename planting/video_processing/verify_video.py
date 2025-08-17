from planting.utils.download_media import download_from_url
from planting.video_processing.extract_frames import extract_frames
from planting.object_detection.detect_objects import detect_video_sequence

def verify_video_planting_from_url(video_url):
    local_path = "planting/uploads/videos/week0.mp4"
    download_from_url(video_url, local_path)
    extract_frames(local_path, "planting/uploads/frames", interval=10)
    return detect_video_sequence("planting/uploads/frames")

