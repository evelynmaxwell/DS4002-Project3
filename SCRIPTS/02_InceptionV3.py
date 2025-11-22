"""
Description:
    Trains an ImageNet-pretrained InceptionV3 CNN to classify
    5 classes of satellite weather imagery, using an 80/10/10 folder split.
    The script builds tf.data pipelines directly from train/val/test directories,
    applies ImageNet preprocessing, fine-tunes the InceptionV3 backbone, and
    evaluates performance on the held-out test set.

Process:
    1. Load train/val/test folders from DATA/dataset_split.
    2. Build tf.data pipelines with preprocessing + batching.
    3. Construct a fully fine-tuned InceptionV3 model.
    4. Train using Adam + early stopping + checkpointing.
    5. Evaluate on test set; print accuracy + confusion matrix metrics.
    6. Save best checkpoint + final model in OUTPUT/.

Inputs:
    - DATA/dataset_split/train/{class_name}/*.jpg
    - DATA/dataset_split/val/{class_name}/*.jpg
    - DATA/dataset_split/test/{class_name}/*.jpg

Outputs:
    - OUTPUT/inceptionv3_best.keras
    - OUTPUT/inceptionv3_final.keras
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


# =====================================================================
#                   CONFIG — REPO-RELATIVE PATHS
# =====================================================================

# Determine repo root dynamically (two levels up from this script)
REPO_ROOT = Path(__file__).resolve().parents[1]

SPLIT_ROOT = REPO_ROOT / "DATA" / "dataset_split"
OUTPUT_DIR = REPO_ROOT / "OUTPUT"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH  = OUTPUT_DIR / "inceptionv3_best.keras"
FINAL_MODEL_PATH = OUTPUT_DIR / "inceptionv3_final.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1
RNG_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


# =====================================================================
def sanity_check_folders(root: Path):
    """Verify that train/val/test folders exist and are non-empty."""
    if not root.exists():
        raise FileNotFoundError(f"dataset_split not found at: {root}")

    split_counts = {}
    for split in ("train", "val", "test"):
        d = root / split
        if not d.exists():
            raise FileNotFoundError(f"Missing split folder: {d}")

        files = [p for p in d.rglob("*") if p.is_file()]
        if len(files) == 0:
            raise ValueError(f"Split folder exists but is empty: {d}")

        split_counts[split] = len(files)

    print("=== Folder sanity check ===")
    for split, cnt in split_counts.items():
        print(f"{split:5s}: {cnt:5d} files")
    print("")


def build_dataset_from_dir(split: str, shuffle: bool, return_class_names=False):
    """Load dataset split and apply preprocessing."""
    split_dir = SPLIT_ROOT / split

    ds_raw = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=RNG_SEED,
    )

    class_names = ds_raw.class_names

    if split == "train":
        print("Detected class names:", class_names)
        print("\nTrain class counts:")
        for cname in class_names:
            count = sum(1 for p in (split_dir / cname).rglob("*") if p.is_file())
            print(f"  {cname:15s}: {count:5d}")
        print("")

    def _preprocess(img, label):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = preprocess_input(img)
        return img, label

    ds = ds_raw.map(_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    if return_class_names:
        return ds, class_names
    return ds


def build_inception_model(num_classes: int) -> tf.keras.Model:
    """Construct a fully fine-tuned InceptionV3 classifier."""
    base = InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    base.trainable = True  # Full fine-tuning

    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


# =====================================================================
def main():
    sanity_check_folders(SPLIT_ROOT)

    print(f"Loading datasets from: {SPLIT_ROOT}\n")
    train_ds, class_names = build_dataset_from_dir("train", True, True)
    val_ds = build_dataset_from_dir("val", False)
    test_ds = build_dataset_from_dir("test", False)

    model = build_inception_model(num_classes=len(class_names))

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    print("\n>>> Starting fine-tuning...\n")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    model.save(FINAL_MODEL_PATH)
    print(f"\nSaved best model to : {BEST_MODEL_PATH}")
    print(f"Saved final model to: {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
