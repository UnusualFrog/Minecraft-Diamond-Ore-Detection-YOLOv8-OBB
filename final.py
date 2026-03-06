import random
import shutil
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

""" ========== Config ========== """
POSITIVE_IMG_DIR = Path("Data/prototype_132/images")
POSITIVE_LBL_DIR = Path("Data/prototype_132/labels")
DATASET_DIR  = Path("Data/dataset")
RUNS_DIR = Path(__file__).parent / "runs" / "obb"
PREDICT_DIR = Path(__file__).parent / "Data" / "predict"

# Train/ Test split
VAL_SPLIT    = 0.2
RANDOM_STATE = 42

PRETRAINED_MODEL = "yolov8n-obb.pt"
EPOCHS           = 100
BATCH_SIZE       = 16
IMG_SIZE         = 640
PATIENCE         = 20
DEVICE           = 0 if torch.cuda.is_available() else "cpu"

""" ========== Load Data ========== """
def build_dataset():
    # Load images
    pos_image_paths = sorted(POSITIVE_IMG_DIR.glob("*.png"))

    # Combine images with YOLO label data
    pos_pairs = []
    for img_path in pos_image_paths:
        lbl_path = POSITIVE_LBL_DIR / (img_path.stem + ".txt")
        pos_pairs.append((img_path, lbl_path))

    # Train Test Split
    random.seed(RANDOM_STATE)
    random.shuffle(pos_pairs)

    # Calculate split as whole number
    n_val       = int(len(pos_pairs) * VAL_SPLIT)
    train_pairs = pos_pairs[n_val:]
    val_pairs   = pos_pairs[:n_val]

    # print(f"Train : {len(train_pairs)}  |  Val : {len(val_pairs)}")

    """ ========== Build Local YOLO-format Dataset ========== """
    # Generate output directories for train and val datasets
    for split in ["train", "val"]:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy train images and labels to respective directories
    for img_path, lbl_path in train_pairs:
        shutil.copy2(img_path, DATASET_DIR / "train" / "images" / img_path.name)
        shutil.copy2(lbl_path, DATASET_DIR / "train" / "labels" / lbl_path.name)

    # Copy test images and labels to respective directories
    for img_path, lbl_path in val_pairs:
        shutil.copy2(img_path, DATASET_DIR / "val" / "images" / img_path.name)
        shutil.copy2(lbl_path, DATASET_DIR / "val" / "labels" / lbl_path.name)

    # Build dataset config file for singular named class "Diamond Ore"
    dataset_cfg = {
        "path"  : str(DATASET_DIR.resolve()),
        "train" : "train/images",
        "val"   : "val/images",
        "nc"    : 1,
        "names" : ["diamond_ore"],
    }

    # Create config file
    yaml_path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path

""" =========== Train Model ==========="""
def train(yaml_path):
    # Load pre-trained 
    model = YOLO(PRETRAINED_MODEL)

    model.train(
        data     = str(yaml_path),
        task     = "obb",
        epochs   = EPOCHS,
        batch    = BATCH_SIZE,
        imgsz    = IMG_SIZE,
        patience = PATIENCE,
        device   = DEVICE,
        plots    = True,
        project  = str(RUNS_DIR),
        name     = "diamond_ore",
    )

    return model

""" ========== Test Model ========== """
def test(yaml_path):
    # Load the weights of the best performing training epoch
    best_weights = RUNS_DIR / "diamond_ore" / "weights" / "best.pt"
    # Re-create model from best weights
    model = YOLO(str(best_weights))

    metrics = model.val(
        data    = str(yaml_path),
        task    = "obb",
        device  = DEVICE,
        plots   = True,
    )

    # Print evaluation metrics
    print(f"\nPrecision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP@50    : {metrics.box.map50:.4f}")
    print(f"mAP@50-95 : {metrics.box.map:.4f}")

""" ========== Predict Un-seen Data =========="""
def predict(path)


def main():
    dataset = build_dataset()
    train(dataset)
    test(dataset)

if __name__ == "__main__":
    main()
