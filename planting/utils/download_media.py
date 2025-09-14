# planting/utils/download_media.py
import os, shutil, requests

def save_uploaded_video(src_path_or_url, dest_folder="uploads/videos"):
    """
    If src_path_or_url is a URL (http/https) it downloads it,
    otherwise copy local file into dest_folder.
    Returns path to saved file.
    """
    os.makedirs(dest_folder, exist_ok=True)
    if isinstance(src_path_or_url, str) and src_path_or_url.startswith(("http://","https://")):
        filename = os.path.basename(src_path_or_url.split("?")[0]) or "downloaded_video.mp4"
        dest = os.path.join(dest_folder, filename)
        with requests.get(src_path_or_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
        return dest
    else:
        # assume local path
        src = src_path_or_url
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source not found: {src}")
        dest = os.path.join(dest_folder, os.path.basename(src))
        if os.path.abspath(src) != os.path.abspath(dest):
            shutil.copy(src, dest)
        return dest
