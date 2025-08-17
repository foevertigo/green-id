import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = (128, 128)
MODEL_PATH = os.path.join("model", "public_transport_model.keras")
CLASS_NAMES_PATH = os.path.join("model", "class_names.pkl")

#  Loading Model
model = tf.keras.models.load_model(MODEL_PATH)

# Loading Class Names
with open(CLASS_NAMES_PATH, 'rb') as f:
    class_names = pickle.load(f)

# Prediction Function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"Predicted Class: {predicted_class} ({confidence * 100:.2f}%)")

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{predicted_class} ({confidence * 100:.2f}%)")
    plt.show()

#  Entry Point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_image")
        sys.exit(1)

    image_path = sys.argv[1]
    
    if not os.path.isfile(image_path):
        print(f"Error: File not found -> {image_path}")
        sys.exit(1)

    predict_image(image_path)
