"""
Filter your YOLO dataset to only baby-safety-relevant classes.

This script:
  1. Reads your existing data.yaml to find the current class mapping
  2. Filters labels to only keep relevant classes
  3. Remaps class IDs to a contiguous 0..N range
  4. Writes a new filtered dataset + data.yaml

Usage:
    python training/filter_dataset.py \
        --input-data path/to/original/data.yaml \
        --output-dir path/to/filtered_dataset

Then retrain with:
    python training/retrain.py --data path/to/filtered_dataset/data.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import yaml
from pathlib import Path


# Classes to KEEP — everything else is dropped.
# These must match EXACTLY the class names in your data.yaml.
KEEP_CLASSES = {
    "person",
    "cat",
    "dog",
    "teddy bear",
    "bottle",
    "book",
    "cup",
    "chair",
    "couch",
    "bed",
    "tv",
    "cell phone",
    "remote",
    "balloon",
    "knife",
    "scissors",
    "fork",
    "power plugs and sockets",
    "bowl",
    "nail",
    "screwdriver",
}


def main():
    parser = argparse.ArgumentParser(description="Filter YOLO dataset to baby-safety classes")
    parser.add_argument("--input-data", required=True, help="Path to original data.yaml")
    parser.add_argument("--output-dir", required=True, help="Output directory for filtered dataset")
    args = parser.parse_args()

    # ── Load original data.yaml ──
    with open(args.input_data, "r") as f:
        data = yaml.safe_load(f)

    original_names = data.get("names", {})
    if isinstance(original_names, list):
        original_names = {i: name for i, name in enumerate(original_names)}

    print(f"Original dataset: {len(original_names)} classes")

    # ── Build old_id → new_id mapping ──
    old_to_new = {}
    new_names = {}
    new_id = 0
    for old_id in sorted(original_names.keys()):
        name = original_names[old_id]
        if name in KEEP_CLASSES:
            old_to_new[old_id] = new_id
            new_names[new_id] = name
            new_id += 1

    print(f"Filtered dataset: {len(new_names)} classes")
    for nid, name in new_names.items():
        print(f"  {nid:2d}. {name}")

    dropped = set(original_names.values()) - KEEP_CLASSES
    print(f"\nDropped {len(dropped)} irrelevant classes:")
    for name in sorted(dropped):
        print(f"  - {name}")

    # ── Process each split (train, val, test) ──
    output_dir = Path(args.output_dir)
    input_base = Path(args.input_data).parent

    for split in ["train", "val", "test"]:
        # Find label/image dirs
        label_dir = input_base / "labels" / split
        image_dir = input_base / "images" / split

        if not label_dir.exists():
            print(f"\nSkipping '{split}' — {label_dir} not found")
            continue

        out_label_dir = output_dir / "labels" / split
        out_image_dir = output_dir / "images" / split
        out_label_dir.mkdir(parents=True, exist_ok=True)
        out_image_dir.mkdir(parents=True, exist_ok=True)

        label_files = list(label_dir.glob("*.txt"))
        kept = 0
        empty = 0

        for lf in label_files:
            lines = lf.read_text().strip().splitlines()
            new_lines = []

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                if old_cls in old_to_new:
                    parts[0] = str(old_to_new[old_cls])
                    new_lines.append(" ".join(parts))

            # Write filtered label (even if empty — means background image)
            out_lf = out_label_dir / lf.name
            out_lf.write_text("\n".join(new_lines) + "\n" if new_lines else "")

            if new_lines:
                kept += 1
            else:
                empty += 1

            # Copy corresponding image
            img_stem = lf.stem
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                img_path = image_dir / f"{img_stem}{ext}"
                if img_path.exists():
                    shutil.copy2(img_path, out_image_dir / img_path.name)
                    break

        print(f"\n{split}: {kept} images with objects, {empty} background-only, {kept + empty} total")

    # ── Write new data.yaml ──
    new_data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(new_names),
        "names": new_names,
    }

    # Add test split if it exists
    if (output_dir / "images" / "test").exists():
        new_data["test"] = "images/test"

    out_yaml = output_dir / "data.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(new_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nFiltered data.yaml written to: {out_yaml}")
    print(f"\nNext step: python training/retrain.py --data {out_yaml}")


if __name__ == "__main__":
    main()
