"""
Description:
  This script performs preprocessing and dataset splitting for a
  5-class satellite weather image dataset. 

Process:
  1. Cleans, normalizes, and standardizes every image.
  2. Removes simple filename-based duplicates.
  3. Resizes all images to 224×224 for CNN ingestion.
  4. Converts all images to RGB and fixes EXIF orientation.
  5. Writes cleaned images to class-specific folders.
  6. Creates deterministic 80/10/10 train/val/test splits.
  7. Saves the split data into a standard structure for model training.

Inputs:
  - DATA/weather_images/{class_name}/*.{jpg,png,tif,...}
  
  Required structure:
      • One folder per weather class.
      • Files may be .jpg/.jpeg/.png/.bmp/.tif/.tiff.
      • Filenames may contain duplicates (detected via “(” “)” or "- Copy").

Outputs:
  - DATA/cleaned_data/{class_name}/*.jpg
        Cleaned, normalized, orientation-corrected, 224×224 RGB images.
  
  - DATA/dataset_split/
        train/{class_name}/*.jpg
        val/{class_name}/*.jpg
        test/{class_name}/*.jpg
"""

from pathlib import Path
from PIL import Image, ImageOps
import shutil
import random

# ======================= CONFIGURATION =======================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "DATA"
RAW_ROOT   = DATA_DIR / "weather_images"

# Where cleaned images will be written
CLEAN_ROOT = DATA_DIR / "cleaned_data"

# Where final train/val/test folders will be written
SPLIT_ROOT = DATA_DIR / "dataset_split"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TARGET_SIZE    = (224, 224)   # model input size
TRAIN_FRAC     = 0.80
VAL_FRAC       = 0.10         # remainder goes to test
RNG_SEED       = 42           # for reproducible splits

# ======================= CHECK INPUT ======================

if not RAW_ROOT.exists():
    raise SystemExit(f"RAW dataset folder not found: {RAW_ROOT}")

# Each immediate subfolder under RAW_ROOT is treated as a class.
class_dirs = sorted([d for d in RAW_ROOT.iterdir() if d.is_dir()])
if not class_dirs:
    raise SystemExit(f"No class subfolders found under: {RAW_ROOT}")

class_names = [d.name for d in class_dirs]

# ================== STEP 1: CLEAN IMAGES ==================

# Start with a fresh cleaned_data/ directory.
if CLEAN_ROOT.exists():
    shutil.rmtree(CLEAN_ROOT)
CLEAN_ROOT.mkdir(parents=True, exist_ok=True)

print("Step 1/2: Preprocessing images...")

for cdir in class_dirs:
    cname = cdir.name
    out_dir = CLEAN_ROOT / cname
    out_dir.mkdir(parents=True, exist_ok=True)

    # Walk all files in this class folder (recursively).
    for src in cdir.rglob("*"):
        if not src.is_file():
            continue

        # Skip unsupported extensions.
        if src.suffix.lower() not in SUPPORTED_EXTS:
            continue

        # Skip obvious duplicate-style names.
        if "(" in src.name or ")" in src.name or "- Copy" in src.stem:
            continue

        try:
            # Open image, fix orientation, convert to RGB.
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")

                # Resize to target size for the CNN.
                im = im.resize(TARGET_SIZE, Image.BILINEAR)

                # Save cleaned image as JPEG in cleaned_data/{class}/.
                out_path = out_dir / f"{src.stem}.jpg"
                im.save(out_path, format="JPEG", quality=95)

        except Exception:
            # If unreadable/corrupted, just skip.
            continue

# =========== STEP 2: CREATE TRAIN / VAL / TEST SPLITS ============

print("Step 2/2: Creating train/val/test splits...")

# Start with a fresh dataset_split/ directory.
if SPLIT_ROOT.exists():
    shutil.rmtree(SPLIT_ROOT)
for split in ("train", "val", "test"):
    (SPLIT_ROOT / split).mkdir(parents=True, exist_ok=True)

random.seed(RNG_SEED)

for cname in class_names:
    class_clean_dir = CLEAN_ROOT / cname

    # List all cleaned images for this class.
    files = sorted([
        p for p in class_clean_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ])

    if not files:
        continue

    # Shuffle deterministically.
    random.shuffle(files)
    n = len(files)

    # Compute split sizes.
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    n_test  = n - n_train - n_val  # remainder

    train_files = files[:n_train]
    val_files   = files[n_train:n_train + n_val]
    test_files  = files[n_train + n_val:]

    # Helper to copy a batch of files into dataset_split/{split}/{class}/.
    def copy_batch(batch, split_name):
        dst_dir = SPLIT_ROOT / split_name / cname
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in batch:
            shutil.copy2(f, dst_dir / f.name)

    copy_batch(train_files, "train")
    copy_batch(val_files,   "val")
    copy_batch(test_files,  "test")

print("Done. Cleaned images and splits are ready.")
print(f"Cleaned data  : {CLEAN_ROOT}")
print(f"Dataset splits: {SPLIT_ROOT}")
