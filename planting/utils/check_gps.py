# planting/utils/check_gps.py
import json, subprocess, re, os

def _extract_gps_from_video_ffprobe(video_path):
    """
    Optional: uses ffprobe to read metadata tags. Requires ffprobe installed.
    Returns string like '+12.3456+098.7654' or None.
    """
    if not os.path.exists(video_path):
        return None
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_entries", "format_tags", video_path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8")
        js = json.loads(out)
        tags = js.get("format", {}).get("tags", {}) or {}
        for key, val in tags.items():
            if "location" in key.lower() or "iso6709" in key.lower():
                return val
        return None
    except Exception:
        return None

def _parse_iso6709(s):
    """
    Parse ISO6709 basic pattern: +12.3456+098.7654
    """
    if not isinstance(s, str):
        return None
    m = re.search(r"([+-]\d+\.\d+)([+-]\d+\.\d+)", s)
    if m:
        return {"lat": float(m.group(1)), "lon": float(m.group(2))}
    return None

def validate_gps(video_path=None, gps_json=None, allowed_bbox=None):
    """
    gps_json: {'lat': float, 'lon': float} preferred.
    allowed_bbox: (min_lat, max_lat, min_lon, max_lon) optional.
    Returns: (True/False, info)
    """
    gps = None
    if gps_json:
        try:
            gps = {"lat": float(gps_json["lat"]), "lon": float(gps_json["lon"])}
        except Exception:
            return False, {"reason": "invalid_gps_format"}
    else:
        if not video_path:
            return False, {"reason": "no_video_or_gps_provided"}
        s = _extract_gps_from_video_ffprobe(video_path)
        if not s:
            return False, {"reason": "no_gps_found_in_video"}
        gps = _parse_iso6709(s)
        if not gps:
            return False, {"reason": "bad_gps_format_in_metadata"}

    if allowed_bbox:
        min_lat, max_lat, min_lon, max_lon = allowed_bbox
        if not (min_lat <= gps["lat"] <= max_lat and min_lon <= gps["lon"] <= max_lon):
            return False, {"reason": "outside_allowed_area", "gps": gps}
    return True, gps

if __name__ == "__main__":
    # quick test
    ok, info = validate_gps(gps_json={"lat": 28.7041, "lon": 77.1025}, allowed_bbox=(20,40,70,90))
    print(ok, info)
