from flask import Flask, request, jsonify
from pathlib import Path
from utils import save_uploaded_file, verify_cleanup

app = Flask(__name__)

# Folders
BASE_DIR = Path(__file__).parent
BEFORE_DIR = BASE_DIR / "uploads" / "before"
AFTER_DIR  = BASE_DIR / "uploads" / "after"

@app.route("/verify_cleanup", methods=["POST"])
def cleanup_check():
    """
    Accepts 'before' and 'after' image files and verifies cleanup.
    Returns JSON with result and points if cleanup verified.
    """
    before_file = request.files.get("before")
    after_file  = request.files.get("after")

    if not before_file or not after_file:
        return jsonify({"error": "Both before and after images are required."}), 400

    # Save uploaded files
    before_path = save_uploaded_file(before_file, BEFORE_DIR)
    after_path  = save_uploaded_file(after_file, AFTER_DIR)

    # Run verification
    result = verify_cleanup(before_path, after_path)
    points = 10 if result else 0

    return jsonify({
        "cleanup_verified": result,
        "points_awarded": points
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
