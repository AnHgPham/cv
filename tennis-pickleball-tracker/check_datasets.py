"""Check all downloaded datasets."""
import os, yaml

datasets = [
    "data/pickleball_vision_v1",
    "data/pickleball_vision_v9",
    "data/pickleball_ball_owl",
    "data/pickleball_ball_tracking",
    "data/pickleball_court_seg",
    "data/pickleball_court_only",
    "data/pickleball_frisbeepb",
    "data/pickleball_pickleball_video",
    "data/pickleball_pb_vision",
    "data/pickleball_ball_player",
]

print(f"{'Dataset':<35} {'Classes':<35} {'Total':<8}")
print("-" * 80)

grand_total = 0
for path in datasets:
    yaml_path = os.path.join(path, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"{os.path.basename(path):<35} {'--- NOT FOUND ---':<35}")
        continue
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    total = 0
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(path, split, "images")
        if os.path.exists(img_dir):
            total += len(os.listdir(img_dir))
    classes = str(cfg.get("names", []))
    name = os.path.basename(path)
    print(f"{name:<35} {classes:<35} {total}")
    grand_total += total

print("-" * 80)
print(f"{'GRAND TOTAL':<35} {'':<35} {grand_total}")
