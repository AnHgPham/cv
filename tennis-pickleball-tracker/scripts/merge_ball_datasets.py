"""
Merge pickleball ball datasets into one unified YOLOv8 dataset.

Combines:
- pickleball_vision_v1 (2,118 imgs): classes [-ball, Court, ball] → keep only ball (id=2 → 0)
- pickleball_vision_v9 (5,390 imgs): classes [ball] → keep as-is (id=0)

Output: data/merged_ball/ with train/valid/test splits, all class ball=0
"""

import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

BASE = Path(r"D:\Downloads\cv\tennis-pickleball-tracker\data")
OUTPUT = BASE / "merged_ball"


def merge_dataset(src_dir: Path, split: str, ball_class_id: int, dst_dir: Path, prefix: str):
    """Copy images and remap labels from src to dst.
    
    Args:
        src_dir: source dataset root
        split: 'train', 'valid', or 'test'
        ball_class_id: class ID of 'ball' in source dataset
        dst_dir: output merged dataset root
        prefix: filename prefix to avoid collisions (e.g., 'v1_', 'v9_') 
    """
    src_img = src_dir / split / "images"
    src_lbl = src_dir / split / "labels"
    dst_img = dst_dir / split / "images"
    dst_lbl = dst_dir / split / "labels"
    
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    
    if not src_img.exists():
        return 0, 0
    
    img_files = list(src_img.iterdir())
    copied = 0
    labels_with_ball = 0
    
    for img_path in img_files:
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            continue
        
        # Copy image with prefix
        dst_img_path = dst_img / f"{prefix}{img_path.name}"
        shutil.copy2(img_path, dst_img_path)
        copied += 1
        
        # Process label file
        lbl_name = img_path.stem + ".txt"
        src_lbl_path = src_lbl / lbl_name
        dst_lbl_path = dst_lbl / f"{prefix}{lbl_name}"
        
        if src_lbl_path.exists():
            # Read and filter/remap labels
            new_lines = []
            with open(src_lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if cls_id == ball_class_id:
                            # Remap to class 0 (ball)
                            new_lines.append(f"0 {' '.join(parts[1:])}")
            
            # Write filtered labels (even if empty - means negative sample)
            with open(dst_lbl_path, 'w') as f:
                f.write('\n'.join(new_lines))
            
            if new_lines:
                labels_with_ball += 1
        else:
            # No label file = negative sample (background)
            with open(dst_lbl_path, 'w') as f:
                pass  # empty file
    
    return copied, labels_with_ball


def main():
    # Clean output
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    
    print("=" * 60)
    print("MERGING PICKLEBALL BALL DATASETS")
    print("=" * 60)
    
    total_imgs = 0
    total_labels = 0
    
    # === Dataset 1: pickleball_vision_v1 ===
    # Classes: {0: '-ball', 1: 'Court', 2: 'ball'}
    # We want class_id=2 (ball) → remap to 0
    src_v1 = BASE / "pickleball_vision_v1"
    print(f"\n[v1] {src_v1}")
    print(f"     Classes: [-ball=0, Court=1, ball=2] → keep ball(2)→0")
    
    for split in ['train', 'valid', 'test']:
        imgs, lbls = merge_dataset(src_v1, split, ball_class_id=2, dst_dir=OUTPUT, prefix="v1_")
        print(f"     {split}: {imgs} images, {lbls} with ball labels")
        total_imgs += imgs
        total_labels += lbls
    
    # === Dataset 2: pickleball_vision_v9 ===
    # Classes: {0: 'ball'}
    # Already class_id=0, keep as-is
    src_v9 = BASE / "pickleball_vision_v9"
    print(f"\n[v9] {src_v9}")
    print(f"     Classes: [ball=0] → keep as-is")
    
    for split in ['train', 'valid', 'test']:
        imgs, lbls = merge_dataset(src_v9, split, ball_class_id=0, dst_dir=OUTPUT, prefix="v9_")
        print(f"     {split}: {imgs} images, {lbls} with ball labels")
        total_imgs += imgs
        total_labels += lbls
    
    # === Create data.yaml ===
    data_yaml = {
        'path': str(OUTPUT.resolve()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['ball'],
    }
    
    yaml_path = OUTPUT / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    # === Summary ===
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"Output: {OUTPUT}")
    print(f"Total images: {total_imgs}")
    print(f"Images with ball labels: {total_labels}")
    print(f"data.yaml: {yaml_path}")
    
    # Count per split
    for split in ['train', 'valid', 'test']:
        img_dir = OUTPUT / split / "images"
        lbl_dir = OUTPUT / split / "labels"
        n_img = len(list(img_dir.iterdir())) if img_dir.exists() else 0
        n_lbl = sum(1 for f in lbl_dir.iterdir() if f.stat().st_size > 0) if lbl_dir.exists() else 0
        print(f"  {split}: {n_img} images, {n_lbl} non-empty labels")
    
    print(f"\ndata.yaml content:")
    print(yaml.dump(data_yaml, default_flow_style=False))


if __name__ == "__main__":
    main()
