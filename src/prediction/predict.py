import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
IMG_SIZE = (224, 224)


def predict_image(crop, image_path):
    model_path = MODELS_DIR / f"{crop}_disease_model.keras"
    classes_path = MODELS_DIR / f"{crop}_classes.json"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    if not classes_path.exists():
        print(f"Classes file not found: {classes_path}")
        sys.exit(1)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    print("Loading model...")

    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")

    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    print("Classes loaded successfully!")

    image = load_img(image_path, target_size=IMG_SIZE)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    print("Running prediction...")

    predictions = model.predict(image_array, verbose=0)

    print("Prediction completed!")

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    print("----------------------------------------")
    print(f"Image:      {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("----------------------------------------")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("python .\\src\\prediction\\predict.py <cassava|maize> <image_path>")
        sys.exit(1)

    crop = sys.argv[1].lower()
    image_path = Path(sys.argv[2])

    if crop not in ["cassava", "maize"]:
        print("Crop must be either 'cassava' or 'maize'.")
        sys.exit(1)

    predict_image(crop, image_path)