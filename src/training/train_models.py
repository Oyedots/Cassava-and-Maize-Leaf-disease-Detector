from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CASSAVA_DIR = PROJECT_ROOT / "data" / "raw" / "cassava" / "dataset" / "train"
MAIZE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "maize"
    / "dataset"
    / "Maize dataset"
)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 10


def create_datasets(data_dir):
    """Create training and validation datasets from class folders."""

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names

    # Improve input pipeline performance
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, class_names


def build_model(num_classes):
    """Build a transfer-learning image classifier."""

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=IMG_SIZE + (3,))

    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_crop(crop_name, data_dir):
    """Train and save a model for one crop."""

    print("\n" + "=" * 60)
    print(f"TRAINING {crop_name.upper()} MODEL")
    print("=" * 60)

    print(f"Dataset: {data_dir}")

    train_ds, val_ds, class_names = create_datasets(data_dir)

    print(f"Classes: {class_names}")

    model = build_model(len(class_names))

    model_path = MODELS_DIR / f"{crop_name}_disease_model.keras"
    labels_path = MODELS_DIR / f"{crop_name}_classes.json"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print("\nTraining complete!")
    print(f"Model saved to: str{model_path}")
    print(f"Classes saved to: {labels_path}")

    return history


if __name__ == "__main__":
    print("Cassava-Maize Leaf Disease Detector")
    print("TensorFlow version:", tf.__version__)

    train_crop("cassava", CASSAVA_DIR)

    train_crop("maize", MAIZE_DIR)

    print("\nAll training completed successfully!")