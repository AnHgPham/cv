"""Download all pickleball datasets from Roboflow.

Usage:
    set ROBOFLOW_API_KEY=VAwW4zaxVs978t7iLszZ
    python download_all_datasets.py
"""
import os
import yaml
from roboflow import Roboflow

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "VAwW4zaxVs978t7iLszZ")
rf = Roboflow(api_key=API_KEY)


def download_and_report(workspace, project, version, fmt, location, desc):
    """Download a Roboflow dataset and report stats."""
    print("=" * 60)
    print(f"Downloading: {desc}")
    print(f"  ws={workspace}  proj={project}  v={version}")
    try:
        p = rf.workspace(workspace).project(project)
        v = p.version(version)
        v.download(fmt, location=location, overwrite=True)

        yaml_path = os.path.join(location, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            print(f"  Classes: {cfg.get('names', [])}")
            print(f"  nc: {cfg.get('nc', 0)}")

        total = 0
        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(location, split, "images")
            if os.path.exists(img_dir):
                n = len(os.listdir(img_dir))
                total += n
                print(f"  {split}: {n} images")
        print(f"  TOTAL: {total} images")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ================================================================
# 1) BALL DETECTION DATASETS
# ================================================================

# 1a) Pickleball Vision v1 (ball + Court, 2118 images)
download_and_report(
    "liberin-technologies", "pickleball-vision", 1,
    "yolov8", "data/pickleball_vision_v1",
    "Pickleball Vision v1 (ball + Court) - Liberin Technologies"
)

# 1b) Pickleball Vision v9 (ball only, 5390 images - largest)
download_and_report(
    "liberin-technologies", "pickleball-vision", 9,
    "yolov8", "data/pickleball_vision_v9",
    "Pickleball Vision v9 (ball only) - Liberin Technologies"
)

# 1c) Pickleball Ball Detection by Owl (499 images)
download_and_report(
    "owl", "pickleball-ball-detection", 1,
    "yolov8", "data/pickleball_ball_owl",
    "Pickleball Ball Detection - Owl"
)

# 1d) Ball Tracking dataset
download_and_report(
    "ball-tracking-glofe", "ball-tracking", 5,
    "yolov8", "data/pickleball_ball_tracking",
    "Ball Tracking v5 - ball tracking project"
)

# ================================================================
# 2) COURT DETECTION DATASETS
# ================================================================

# 2a) Pickleball Court Segmentation (120 images)
download_and_report(
    "gideons", "pickleball-court", 1,
    "yolov8", "data/pickleball_court_seg",
    "Pickleball Court Segmentation - Gideons"
)

# ================================================================
# 3) PLAYER/PERSON DETECTION
# ================================================================
# Note: For player detection, YOLO pretrained on COCO already
# detects 'person' class. Additional pickleball player datasets
# can be searched on Roboflow Universe.

# Try some known pickleball projects with player annotations
player_datasets = [
    ("frisbeepb", "frisbeepb", 1, "Frisbee/PB with player"),
    ("my-project-wmzzv", "pickleball-video", 1, "Pickleball Video"),
    ("pbvision", "pb-vision", 1, "PB Vision"),
]

for ws, proj, ver, desc in player_datasets:
    download_and_report(
        ws, proj, ver,
        "yolov8", f"data/pickleball_{proj.replace('-','_')}",
        f"Player Detection: {desc}"
    )

print("\n" + "=" * 60)
print("ALL DOWNLOADS COMPLETE")
print("=" * 60)
