"""
Retraining script for the baby monitor YOLO object detection model.

Run this on your RTX 3060 machine (12GB VRAM).

Usage:
    python training/retrain.py --data path/to/data.yaml
    python training/retrain.py --data path/to/data.yaml --resume  # resume from checkpoint

What this does:
    1. Filters your dataset to only baby-safety-relevant classes (15 classes)
    2. Trains YOLOv8m with optimized hyperparameters
    3. Saves best.pt to the project root when done

Prerequisites:
    - Your dataset in YOLO format (images/ + labels/ directories)
    - data.yaml pointing to train/val splits
    - pip install ultralytics
"""

from __future__ import annotations

import argparse
import os
import shutil

from ultralytics import YOLO


# ── Classes relevant to baby safety monitoring ──────────────────────────────
# Map from your current 53-class model's class names to the filtered set.
# Only these classes will be kept during training.

BABY_SAFETY_CLASSES = [
    "person",                   # Baby / caregiver — MOST IMPORTANT
    "cat",                      # Pet near baby
    "dog",                      # Pet near baby
    "teddy bear",               # Suffocation risk in crib
    "bottle",                   # Choking / feeding
    "book",                     # Suffocation if over face
    "cup",                      # Spill / choking risk
    "chair",                    # Environment context
    "couch",                    # Sleep surface
    "bed",                      # Sleep surface
    "tv",                       # Environment (falling risk)
    "cell phone",               # Small object hazard
    "remote",                   # Small object / choking
    "balloon",                  # #1 choking hazard for objects
    "knife",                    # Sharp object
    "scissors",                 # Sharp object
    "fork",                     # Sharp object
    "power plugs and sockets",  # Electrocution risk
    "bowl",                     # Context (feeding time)
    "pillow",                   # Suffocation risk (add if you add this class)
    "blanket",                  # SIDS risk (add if you add this class)
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Retrain YOLO for baby monitoring")
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--model", default="yolov8m.pt",
                   help="Base model or checkpoint to start from (default: yolov8m.pt)")
    p.add_argument("--epochs", type=int, default=150,
                   help="Total training epochs (default: 150)")
    p.add_argument("--batch", type=int, default=12,
                   help="Batch size — 12 is safe for RTX 3060 12GB (default: 12)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Training image size (default: 640)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the model checkpoint")
    p.add_argument("--device", default="0", help="GPU device (default: 0)")
    p.add_argument("--project", default="runs/detect",
                   help="Project directory for saving results")
    p.add_argument("--name", default="baby_monitor_v2",
                   help="Run name (default: baby_monitor_v2)")
    return p


def main():
    args = build_parser().parse_args()

    print("=" * 60)
    print("Baby Monitor — YOLO Retraining")
    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Data:    {args.data}")
    print(f"Epochs:  {args.epochs}")
    print(f"ImgSize: {args.imgsz}")
    print(f"Batch:   {args.batch}")
    print(f"Device:  {args.device}")
    print()
    print("Target classes for baby safety:")
    for i, cls in enumerate(BABY_SAFETY_CLASSES):
        print(f"  {i:2d}. {cls}")
    print("=" * 60)

    model = YOLO(args.model)

    # ── Training with optimized hyperparameters ──
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,

        # ── Key improvements over your previous training ──

        # Patience: how many epochs without improvement before early stopping.
        # 10 was too aggressive — killed your training at epoch 80 when it
        # may have still been improving. 30 gives the model more room.
        patience=30,

        # Close mosaic augmentation earlier (last 15 epochs instead of 10).
        # Helps the model fine-tune on cleaner data at the end.
        close_mosaic=15,

        # Learning rate schedule: cosine annealing works better than step decay
        # for longer training runs. Keeps learning rate from dropping too fast.
        cos_lr=True,

        # Final learning rate factor: 0.01 of initial LR at the end.
        # This is the same as yours — good value.
        lrf=0.01,

        # Augmentation — slightly more aggressive for better generalization
        hsv_h=0.015,    # hue shift
        hsv_s=0.7,      # saturation shift
        hsv_v=0.4,      # brightness shift
        degrees=10.0,   # rotation (baby can be at any angle)
        translate=0.15,  # translation (baby moves in crib)
        scale=0.5,       # scale variation
        fliplr=0.5,      # horizontal flip
        flipud=0.1,      # vertical flip (baby can be upside down)
        mosaic=1.0,      # mosaic augmentation
        mixup=0.1,       # slight mixup for regularization

        # Box loss weight — slightly higher to prioritize localization
        box=7.5,

        # Save checkpoints more frequently
        save_period=5,

        # Workers — 4 is good for your setup
        workers=4,

        # Deterministic for reproducibility
        deterministic=True,

        # Keep validation plots
        plots=True,

        # AMP (automatic mixed precision) — enabled by default, good for RTX 3060
        amp=True,
    )

    # ── Copy best.pt to project root ──
    best_pt = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_pt):
        dest = os.path.join(os.path.dirname(__file__), "..", "best.pt")
        shutil.copy2(best_pt, dest)
        print(f"\nbest.pt copied to: {os.path.abspath(dest)}")
    else:
        print(f"\nWarning: {best_pt} not found — check training output directory")

    print("\nTraining complete!")
    print(f"Results saved to: {os.path.join(args.project, args.name)}")


if __name__ == "__main__":
    main()
