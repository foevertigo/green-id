# planting/verification_pipeline.py
from fastapi import FastAPI, UploadFile, File, Form
import shutil, os, json
from pathlib import Path

from utils.download_media import save_uploaded_video
from utils.check_gps import validate_gps
from utils.cleanup import ensure_empty_dir, cleanup_paths
from video_processing.extract_frames import extract_frames
from video_processing.verify_video import verify_planting_from_frames

app = FastAPI()
BASE = Path(__file__).parent.resolve()
UPLOADS_VIDEOS = BASE / ".." / "uploads" / "videos"
UPLOADS_FRAMES = BASE / ".." / "uploads" / "frames"
# normalize
UPLOADS_VIDEOS = UPLOADS_VIDEOS.resolve()
UPLOADS_FRAMES = UPLOADS_FRAMES.resolve()
os.makedirs(UPLOADS_VIDEOS, exist_ok=True)
os.makedirs(UPLOADS_FRAMES, exist_ok=True)

@app.get("/")
def root():
    return {"msg": "Green-ID planting verification service running"}

@app.post("/verify")
async def verify_endpoint(file: UploadFile = File(None), lat: float = Form(None), lon: float = Form(None), use_demo: bool = Form(False)):
    """
    POST /verify
    - file : video file (optional if use_demo True)
    - lat, lon : optional GPS provided by frontend (preferred)
    - use_demo: if True, uses uploads/videos/demo_video.mp4 instead of uploaded file
    """
    # Decide source video
    if use_demo:
        video_path = (BASE / ".." / "uploads" / "videos" / "demo_video.mp4").resolve()
        if not video_path.exists():
            return {"status":"fail","reason":"demo_video_missing"}
    else:
        if file is None:
            return {"status":"fail","reason":"no_file_provided"}
        saved = save_uploaded_video(file.file if hasattr(file, "file") else file.filename, dest_folder=str(UPLOADS_VIDEOS))
        # save_uploaded_video handles paths or file-like? we have UploadFile - save explicitly
        # For UploadFile, write directly:
        if hasattr(file, "filename"):
            video_path = UPLOADS_VIDEOS / file.filename
            with open(video_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        else:
            return {"status":"fail","reason":"invalid_file"}

    # Validate GPS (prefer lat/lon passed)
    gps_json = None
    if lat is not None and lon is not None:
        gps_json = {"lat": lat, "lon": lon}
    ok, gps_info = validate_gps(video_path=str(video_path), gps_json=gps_json, allowed_bbox=( -90, 90, -180, 180))
    if not ok:
        # still continue optional: return fail now
        return {"status":"fail", "reason":"gps_failed", "info": gps_info}

    # Extract frames
    ensure_empty_dir(str(UPLOADS_FRAMES))
    frames = extract_frames(str(video_path), out_dir=str(UPLOADS_FRAMES), sample_fps=1, max_frames=60)
    if not frames:
        cleanup_paths([str(UPLOADS_FRAMES)])
        return {"status":"fail","reason":"no_frames_extracted"}

    # Verify planting
    passed, evidence = verify_planting_from_frames(str(UPLOADS_FRAMES), min_plant_frames=1, motion_threshold=0.6)

    # Cleanup frames (keep video)
    cleanup_paths([str(UPLOADS_FRAMES)])

    response = {
        "status": "success" if passed else "fail",
        "passed": bool(passed),
        "evidence": evidence,
        "gps": gps_info
    }
    if passed:
        response["points_awarded"] = 10
    return response
