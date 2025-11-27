# S:\BTech\Labmentix\Aerial-Object-Detection\detection\train_yolo.py

import os
import torch

# ------------------------------------------------------------------
# 0. PATCH torch.load TO DISABLE weights_only=True BEHAVIOR
# ------------------------------------------------------------------
# PyTorch 2.6+ changed default weights_only=True, which breaks older
# Ultralytics checkpoints. We trust our checkpoints, so we force
# weights_only=False globally.

_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # Force weights_only=False for all calls unless user explicitly overrides
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Now import Ultralytics AFTER patching torch.load
from ultralytics import YOLO


def main():
    # -----------------------------
    # 1. Basic paths and checks
    # -----------------------------
    ROOT = r"S:\BTech\Labmentix\Aerial-Object-Detection\detection"
    DATA_YAML = os.path.join(ROOT, "bird_drone.yaml")

    print("======================================")
    print("YOLOv8 Bird/Drone DETECTION Training")
    print("ROOT        :", ROOT)
    print("DATA_YAML   :", DATA_YAML)
    print("======================================")

    print("YAML exists       :", os.path.exists(DATA_YAML))
    print("Train dir exists  :", os.path.isdir(os.path.join(ROOT, "train", "images")))
    print("Val dir exists    :", os.path.isdir(os.path.join(ROOT, "valid", "images")))
    print("Test dir exists   :", os.path.isdir(os.path.join(ROOT, "test", "images")))
    print("")

    # -----------------------------
    # 2. Try to load previous best, else fallback to yolov8s.pt
    # -----------------------------
    prev_best = os.path.join(
        ROOT,
        "runs", "detect", "bird_drone_det9", "weights", "best.pt"
    )

    model = None

    if os.path.exists(prev_best):
        print("✅ Found previous best detector:")
        print("   ", prev_best)
        print("   -> Attempting to FINE-TUNE this model...\n")
        try:
            model = YOLO(prev_best)
            print("✅ Successfully loaded previous best.pt\n")
        except Exception as e:
            print("⚠ Failed to load previous best.pt due to:")
            print("   ", repr(e))
            print("   -> Falling back to base YOLOv8s weights instead.\n")

    if model is None:
        base_weights = "yolov8s.pt"
        print("Using base model:", base_weights)
        model = YOLO(base_weights)

    # -----------------------------
    # 3. Train with strong augmentations
    # -----------------------------
    print("======================================")
    print("Starting training...")
    print("======================================")

    model.train(
        data=DATA_YAML,
        epochs=60,             # adjust (40–80) if you want
        imgsz=640,
        batch=8,
        device="cpu",          # change to "0" if you get a GPU
        name="bird_drone_det_finetune_aug",

        # ---------- DATA AUGMENTATION ----------
        # Color / brightness / contrast
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Geometric transforms: rotation, translation, zoom (scale), shear
        degrees=15.0,
        translate=0.10,
        scale=0.5,
        shear=2.0,
        perspective=0.0,

        # Flips
        flipud=0.1,
        fliplr=0.5,

        # Mix-style augmentations
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1,

        # Disable mosaic in last 10 epochs
        close_mosaic=10,

        # Windows-safe
        workers=0,
    )

    # -----------------------------
    # 4. Validate
    # -----------------------------
    print("\n======================================")
    print("Running validation...")
    print("======================================")
    model.val(data=DATA_YAML, device="cpu")

    # -----------------------------
    # 5. Predict on test images
    # -----------------------------
    print("\n======================================")
    print("Running predictions on test set...")
    print("======================================")

    test_images_dir = os.path.join(ROOT, "test", "images")
    model.predict(
        source=test_images_dir,
        save=True,
        conf=0.25,
        device="cpu"
    )

    print("\n✅ Training + validation + predictions completed.")


if __name__ == "__main__":
    main()
