from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path

app = FastAPI(title="Cassava-Maize Leaf Disease Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

IMG_SIZE = (224, 224)

models = {}
class_names = {}

for crop in ["cassava", "maize"]:
    model_path = MODELS_DIR / f"{crop}_saved_model"
    classes_path = MODELS_DIR / f"{crop}_classes.json"

    if model_path.exists() and classes_path.exists():
        models[crop] = tf.saved_model.load(str(model_path))

        with open(classes_path, "r", encoding="utf-8") as f:
            class_names[crop] = json.load(f)

        print(f"{crop.upper()} MODEL LOADED")


@app.get("/")
def home():
    return {
        "message": "Cassava-Maize Leaf Disease Detector API",
        "status": "running",
        "models": list(models.keys()),
    }


@app.post("/predict/{crop}")
async def predict(crop: str, file: UploadFile = File(...)):
    crop = crop.lower()

    if crop not in models:
        return {"error": f"Model not available for {crop}"}

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    model = models[crop]

    predictions = model.signatures["serving_default"](
        tf.constant(image_array, dtype=tf.float32)
    )

    output = list(predictions.values())[0]

    predicted_index = int(np.argmax(output.numpy()[0]))
    predicted_class = class_names[crop][predicted_index]
    confidence = float(output.numpy()[0][predicted_index]) * 100

    return {
        "crop": crop,
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
    }