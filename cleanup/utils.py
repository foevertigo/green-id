import os
import logging
from ultralytics import YOLO
from pathlib import Path
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load model once
model_path = Path(__file__).parent / "model" / "best.pt"
model = YOLO(model_path)
# Class mapping
cls_map = {0: "after", 1: "before"}
def verify_cleanup(before_img_path, after_img_path, confidence_threshold=0.5, log_details=True):
    """
    Verify cleanup with improved confidence threshold and detailed logging.
    
    Args:
        before_img_path: Path to the 'before' cleanup image
        after_img_path: Path to the 'after' cleanup image
        confidence_threshold: Minimum confidence required for verification (default: 0.5)
        log_details: Whether to log detailed prediction information
    
    Returns:
        dict: {
            'verified': bool,
            'reason': str,
            'before_class': str,
            'before_confidence': float,
            'after_class': str,
            'after_confidence': float,
            'before_probs': dict,
            'after_probs': dict
        }
    """
    # Predict both images
    res_before = model.predict(before_img_path, verbose=False)[0].probs
    res_after = model.predict(after_img_path, verbose=False)[0].probs
    
    # Get predictions and confidences
    before_class = cls_map[res_before.top1]
    before_conf = res_before.top1conf.item()
    
    after_class = cls_map[res_after.top1]
    after_conf = res_after.top1conf.item()
    
    # Get full probability distributions
    before_probs = {
        'after': res_before.data[0].item(),
        'before': res_before.data[1].item()
    }
    
    after_probs = {
        'after': res_after.data[0].item(),
        'before': res_after.data[1].item()
    }
    
    # Log details if requested
    if log_details:
        logger.info(f"Before image: {before_img_path}")
        logger.info(f"  Predicted: {before_class} (confidence: {before_conf:.2%})")
        logger.info(f"  Probabilities - After: {before_probs['after']:.2%}, Before: {before_probs['before']:.2%}")
        
        logger.info(f"After image: {after_img_path}")
        logger.info(f"  Predicted: {after_class} (confidence: {after_conf:.2%})")
        logger.info(f"  Probabilities - After: {after_probs['after']:.2%}, Before: {after_probs['before']:.2%}")
    
    # Check confidence thresholds
    if before_conf < confidence_threshold:
        reason = f"Before image confidence too low ({before_conf:.2%} < {confidence_threshold:.2%})"
        if log_details:
            logger.warning(reason)
        
        return {
            'verified': False,
            'reason': reason,
            'before_class': before_class,
            'before_confidence': before_conf,
            'after_class': after_class,
            'after_confidence': after_conf,
            'before_probs': before_probs,
            'after_probs': after_probs
        }
    
    if after_conf < confidence_threshold:
        reason = f"After image confidence too low ({after_conf:.2%} < {confidence_threshold:.2%})"
        if log_details:
            logger.warning(reason)
        
        return {
            'verified': False,
            'reason': reason,
            'before_class': before_class,
            'before_confidence': before_conf,
            'after_class': after_class,
            'after_confidence': after_conf,
            'before_probs': before_probs,
            'after_probs': after_probs
        }
    
    # Verify cleanup: before should be "before" and after should be "after"
    verified = (before_class == "before" and after_class == "after")
    
    if verified:
        reason = "Cleanup verified successfully"
        if log_details:
            logger.info(reason)
    else:
        reason = f"Classification mismatch: before={before_class}, after={after_class}"
        if log_details:
            logger.warning(reason)
    
    return {
        'verified': verified,
        'reason': reason,
        'before_class': before_class,
        'before_confidence': before_conf,
        'after_class': after_class,
        'after_confidence': after_conf,
        'before_probs': before_probs,
        'after_probs': after_probs
    }
def save_uploaded_file(file, folder):
    """Save uploaded image to folder, return saved path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file.filename)
    file.save(path)
    return path