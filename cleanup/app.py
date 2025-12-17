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
    Returns JSON with detailed verification results.
    
    Optional parameters:
    - confidence_threshold: Minimum confidence for verification (default: 0.5)
    """
    before_file = request.files.get("before")
    after_file  = request.files.get("after")
    if not before_file or not after_file:
        return jsonify({"error": "Both before and after images are required."}), 400
    # Get optional confidence threshold parameter
    confidence_threshold = float(request.form.get("confidence_threshold", 0.5))
    
    # Validate threshold
    if not 0.0 <= confidence_threshold <= 1.0:
        return jsonify({"error": "confidence_threshold must be between 0.0 and 1.0"}), 400
    # Save uploaded files
    before_path = save_uploaded_file(before_file, BEFORE_DIR)
    after_path  = save_uploaded_file(after_file, AFTER_DIR)
    # Run verification with improved function
    result = verify_cleanup(
        before_path, 
        after_path, 
        confidence_threshold=confidence_threshold,
        log_details=True
    )
    
    # Award points only if cleanup is verified
    points = 10 if result['verified'] else 0
    return jsonify({
        "cleanup_verified": result['verified'],
        "points_awarded": points,
        "reason": result['reason'],
        "details": {
            "before": {
                "predicted_class": result['before_class'],
                "confidence": round(result['before_confidence'], 4),
                "probabilities": {
                    "after": round(result['before_probs']['after'], 4),
                    "before": round(result['before_probs']['before'], 4)
                }
            },
            "after": {
                "predicted_class": result['after_class'],
                "confidence": round(result['after_confidence'], 4),
                "probabilities": {
                    "after": round(result['after_probs']['after'], 4),
                    "before": round(result['after_probs']['before'], 4)
                }
            }
        }
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)