import random
import shutil
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

""" ========== Config ========== """
# Location of YOLO format images and labels
POSITIVE_IMG_DIR = Path("Data/final_dataset_600/images")
POSITIVE_LBL_DIR = Path("Data/final_dataset_600/labels")
# Location to build YOLO dataset
DATASET_DIR  = Path("Data/dataset")
# Location of output
RUNS_DIR = Path(__file__).parent / "runs" / "obb"
# Location of unseen data for predictions
PREDICT_DIR = Path(__file__).parent / "Data" / "predict"
# Utilize pre-trained YOLOv8 w/ OBB model
PRETRAINED_MODEL = "yolov8s-obb.pt"

# Global Random Seed 42 for reproducability
RANDOM_STATE = 42
# Train/ Test split of 80/20
VAL_SPLIT = 0.2
# Use GPU if available, otherwise use CPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  

# Training parameters
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
PATIENCE = 20 # Number of epochs to re-try with no change before early termination


""" ========== Load Data ========== """
def build_dataset():
    # Use existing dataset if exists
    if (DATASET_DIR / "dataset.yaml").exists():
        print(f"Using existing dataset at {DATASET_DIR}")
        return DATASET_DIR / "dataset.yaml"

    # Load images
    pos_image_paths = sorted(POSITIVE_IMG_DIR.glob("*.png"))

    # Combine images with YOLO label data
    pos_pairs = []
    for img_path in pos_image_paths:
        lbl_path = POSITIVE_LBL_DIR / (img_path.stem + ".txt")

        # Ensure label path exists
        if lbl_path.exists():
            pos_pairs.append((img_path, lbl_path))
        else:
            print(f"Warning: missing label for {img_path.name}, skipping.")

    # Train Test Split
    random.seed(RANDOM_STATE)
    random.shuffle(pos_pairs)

    # Calculate dynamic train/test split as whole number based on sample size
    n_val       = int(len(pos_pairs) * VAL_SPLIT)
    train_pairs = pos_pairs[n_val:]
    val_pairs   = pos_pairs[:n_val]

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

# Get best performing model weights
def get_best_weights(run_dir):
    weights = run_dir / "weights" / "best.pt"

    # Check if best.pt exists
    if not weights.exists():
        raise FileNotFoundError(f"No best.pt found at {weights}. Did training complete?")
    
    return weights

""" =========== Train Model ==========="""
def train(yaml_path):
    # Load pre-trained 
    model = YOLO(PRETRAINED_MODEL)

    model.train(
        data = str(yaml_path),
        task = "obb",           # Use oriented bounding boxes (not straight)
        epochs = EPOCHS,
        batch = BATCH_SIZE,
        imgsz = IMG_SIZE,
        patience = PATIENCE,
        device = DEVICE,
        plots = True,            # Generate images with bounding boxes plotting predictions
        project = str(RUNS_DIR),   
        name = "diamond_ore"
    )

    run_dir = Path(model.trainer.save_dir)
    return run_dir

""" ========== Test Model ========== """
def test(yaml_path, run_dir):
    # Load the weights of the model for the best performing training epoch
    model = YOLO(str(get_best_weights(run_dir)))

    metrics = model.val(
        data = str(yaml_path),
        task = "obb",
        device = DEVICE,
        plots = True,
        project = str(RUNS_DIR),
        name = "validate",
        exist_ok= True
    )

    # Print evaluation metrics
    print(f"\nPrecision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP@50    : {metrics.box.map50:.4f}")
    print(f"mAP@50-95 : {metrics.box.map:.4f}")

""" ========== Predict Un-seen Data =========="""
def predict(run_dir):
    # Reconstruct model using best weights
    model = YOLO(str(get_best_weights(run_dir)))

    results = model.predict(
        source = str(PREDICT_DIR),
        task = "obb",
        device = DEVICE,
        conf = 0.25,             # Predictions below conf threshold will not be counted
        save = True,             
        project = str(RUNS_DIR),
        name = "predict",
    )

    # Print number of occurances of target in each sample
    for result in results:
        boxes = result.obb
        n = len(boxes)
        filename = Path(result.path).name

        # No occurences
        if n == 0:
            print(f"{filename} — no detections")
            continue
        
        # Get model confidence score
        confs = boxes.conf.cpu().numpy()
        print(f"{filename} — {n} detection(s) | "
              f"Avg. Confidence: {confs.mean():.2f}")

def main():
    print(f"Using device: {DEVICE}")
    if DEVICE != "cpu":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Construct YOLO dataset
    dataset = build_dataset()
    run_dir = None
    
    while True:
        print("Select an option")
        print("0. Exit")
        print("1. Train Model")
        print("2. Test Model")
        print("3. Predict Unseen Data")
        res = input()

        if res == "0":
            break
        elif res == "1":
            run_dir = train(dataset)
        elif res == "2":
            if run_dir is None:
                # Fall back to most recent existing run
                run_dir = RUNS_DIR / "diamond_ore"
            test(dataset, run_dir)
        elif res == "3":
            if run_dir is None:
                # Fall back to most recent existing run 
                run_dir = RUNS_DIR / "diamond_ore"
            predict(run_dir)
        else:
            print("Error: Invalid Input")
    

if __name__ == "__main__":
    main()
