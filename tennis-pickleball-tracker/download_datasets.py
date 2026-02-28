"""Download pickleball datasets from Roboflow."""
import os
import yaml
from roboflow import Roboflow

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("Please set ROBOFLOW_API_KEY environment variable. "
                     "Example: set ROBOFLOW_API_KEY=your_key_here")
rf = Roboflow(api_key=API_KEY)

# 1) Pickleball Vision v1 (Court + Ball detection)
print("=" * 50)
print("Downloading Pickleball Vision v1 (Court + Ball)...")
project1 = rf.workspace("liberin-technologies").project("pickleball-vision")
v1 = project1.version(1)
ds1 = v1.download("yolov8", location="data/pickleball_vision_v1", overwrite=True)

with open("data/pickleball_vision_v1/data.yaml") as f:
    cfg1 = yaml.safe_load(f)
print(f"  Classes: {cfg1.get('names', [])}")
print(f"  nc: {cfg1.get('nc', 0)}")

# Count files
for split in ["train", "valid", "test"]:
    img_dir = os.path.join("data/pickleball_vision_v1", split, "images")
    if os.path.exists(img_dir):
        n = len(os.listdir(img_dir))
        print(f"  {split}: {n} images")

# 2) Pickleball-Court segmentation (120 images)
print("\n" + "=" * 50)
print("Downloading pickleball-court segmentation...")
try:
    project2 = rf.workspace("gideons").project("pickleball-court")
    v2 = project2.version(1)
    ds2 = v2.download("yolov8", location="data/pickleball_court_seg", overwrite=True)

    with open("data/pickleball_court_seg/data.yaml") as f:
        cfg2 = yaml.safe_load(f)
    print(f"  Classes: {cfg2.get('names', [])}")
    print(f"  nc: {cfg2.get('nc', 0)}")
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join("data/pickleball_court_seg", split, "images")
        if os.path.exists(img_dir):
            n = len(os.listdir(img_dir))
            print(f"  {split}: {n} images")
except Exception as e:
    print(f"  Failed: {e}")

print("\nDone!")
