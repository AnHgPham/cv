"""Merge court segmentation datasets for YOLOv8-seg training."""
import os
import shutil
import yaml
from pathlib import Path

BASE = Path(r"D:\Downloads\cv\tennis-pickleball-tracker\data")
OUTPUT = BASE / "merged_court"

DATASETS = [
    {
        "name": "pickleball_court_only",
        "path": BASE / "pickleball_court_only",
        "court_class_id": 0,  # Court = class 0
    },
    {
        "name": "pickleball_court_seg",
        "path": BASE / "pickleball_court_seg",
        "court_class_id": 0,  # Court = class 0
    },
]


def merge():
    print("=" * 60)
    print("MERGING COURT SEGMENTATION DATASETS")
    print("=" * 60)

    # Clean output
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    for split in ["train", "valid", "test"]:
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {}

    for ds in DATASETS:
        name = ds["name"]
        path = ds["path"]
        court_id = ds["court_class_id"]

        if not path.exists():
            print(f"[SKIP] {name}: not found at {path}")
            continue

        # Read data.yaml
        yaml_path = path / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            classes = cfg.get("names", [])
            print(f"\n[{name}] Classes: {classes}")
        else:
            print(f"\n[{name}] No data.yaml")

        for split in ["train", "valid", "test"]:
            img_src = path / split / "images"
            lbl_src = path / split / "labels"

            if not img_src.exists():
                continue

            imgs = list(img_src.iterdir())
            copied = 0
            seg_labels = 0

            for img_file in imgs:
                # Prefix to avoid name collisions
                prefix = name.replace("pickleball_", "")
                new_name = f"{prefix}_{img_file.name}"

                # Copy image
                dst_img = OUTPUT / split / "images" / new_name
                shutil.copy2(img_file, dst_img)

                # Process label
                lbl_file = lbl_src / (img_file.stem + ".txt")
                dst_lbl = OUTPUT / split / "labels" / (f"{prefix}_{img_file.stem}.txt")

                if lbl_file.exists():
                    lines = lbl_file.read_text().strip().split("\n")
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_id = int(parts[0])
                        if cls_id == court_id:
                            # Remap to class 0 and keep polygon coords
                            new_lines.append("0 " + " ".join(parts[1:]))
                    if new_lines:
                        dst_lbl.write_text("\n".join(new_lines) + "\n")
                        seg_labels += 1
                    else:
                        # Empty label (no court in this image)
                        dst_lbl.write_text("")
                else:
                    dst_lbl.write_text("")

                copied += 1

            print(f"  {split}: {copied} images, {seg_labels} with court seg")
            stats[f"{name}_{split}"] = (copied, seg_labels)

    # Create data.yaml
    data_yaml = {
        "path": str(OUTPUT),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["Court"],
    }
    yaml_path = OUTPUT / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # Summary
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    for split in ["train", "valid", "test"]:
        img_dir = OUTPUT / split / "images"
        lbl_dir = OUTPUT / split / "labels"
        n_imgs = len(list(img_dir.iterdir())) if img_dir.exists() else 0
        n_lbls = sum(1 for f in lbl_dir.iterdir() if f.stat().st_size > 0) if lbl_dir.exists() else 0
        print(f"  {split}: {n_imgs} images, {n_lbls} with court labels")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    merge()
