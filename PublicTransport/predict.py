import sys
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- CONFIG ----------------
# For transfer learning model (MobileNetV2)
IMG_SIZE_TRANSFER = (224, 224)

# For basic CNN model
IMG_SIZE_BASIC = (128, 128)

MODEL_PATH = os.path.join(
    "model",
    "public_transport_model"  # SavedModel directory (no extension)
)

# Classes MUST match folder names (alphabetical order used by Keras)
CLASS_NAMES = [
    "auto_rickshaw",
    "bus",
    "metro",
    "not_transport"
]
# ----------------------------------------


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    print("Loading model...")
    
    # Keras 3 requires using TFSMLayer for SavedModel format
    try:
        # Try loading as Keras model first (for .keras or .h5 files)
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✓ Model loaded successfully")
        return model
    except ValueError:
        # If that fails, it's a SavedModel - wrap it in TFSMLayer for Keras 3
        print("Loading SavedModel with TFSMLayer (Keras 3)...")
        
        # Load the SavedModel as a layer
        tfsm_layer = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')
        
        # Create a wrapper model
        # Determine input shape from the SavedModel
        inputs = tf.keras.Input(shape=(224, 224, 3))  # MobileNetV2 default
        outputs = tfsm_layer(inputs)
        
        # The output might be a dictionary, extract the actual predictions
        if isinstance(outputs, dict):
            # Get the first output (usually the predictions)
            outputs = list(outputs.values())[0]
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        print("✓ Model loaded successfully (SavedModel with TFSMLayer)")
        return model


def detect_model_type(model):
    """
    Detect if the model is transfer learning (MobileNetV2) or basic CNN
    """
    # Check if model contains MobileNetV2 layers
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower() or 'tfsm' in layer.name.lower():
            return 'transfer', IMG_SIZE_TRANSFER
    
    return 'basic', IMG_SIZE_BASIC


def predict_image_basic(img_path, model, img_size):
    """Prediction for basic CNN model"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    return predictions


def predict_image_transfer(img_path, model, img_size):
    """Prediction for transfer learning model (MobileNetV2)"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # MobileNetV2 preprocessing
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    return predictions


def predict_image(img_path, model):
    """
    Auto-detect model type and use appropriate preprocessing
    """
    model_type, img_size = detect_model_type(model)
    
    print(f"Detected model type: {model_type}")
    print(f"Using image size: {img_size}")
    
    if model_type == 'transfer':
        predictions = predict_image_transfer(img_path, model, img_size)
    else:
        predictions = predict_image_basic(img_path, model, img_size)
    
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    return CLASS_NAMES[predicted_index], confidence, predictions[0]


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    model = load_model()
    label, confidence, all_predictions = predict_image(image_path, model)

    print("\nPrediction Result")
    print("="*60)
    print(f"Class      : {label}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print("\n" + "="*60)
    print("All Class Probabilities:")
    print("="*60)
    for i, class_name in enumerate(CLASS_NAMES):
        prob = all_predictions[i] * 100
        bar_length = int(prob / 2)  # Scale to 50 chars max
        bar = "█" * bar_length
        print(f"{class_name:20s}: {prob:6.2f}% {bar}")
    print("="*60)