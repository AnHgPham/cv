import os, yaml

base = r"D:\Downloads\cv\tennis-pickleball-tracker\data"
datasets = {
    "pickleball_court_only": os.path.join(base, "pickleball_court_only"),
    "pickleball_court_seg": os.path.join(base, "pickleball_court_seg"),
    "pickleball_vision_v1": os.path.join(base, "pickleball_vision_v1"),
}

for name, path in datasets.items():
    if not os.path.exists(path):
        print(f"{name}: NOT FOUND")
        continue
    yaml_path = os.path.join(path, "data.yaml")
    classes = "N/A"
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            d = yaml.safe_load(f)
        classes = d.get("names", "N/A")
    counts = {}
    is_seg = False
    sample = ""
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(path, split, "images")
        if os.path.exists(img_dir):
            counts[split] = len(os.listdir(img_dir))
        lbl_dir = os.path.join(path, split, "labels")
        if os.path.exists(lbl_dir):
            files = os.listdir(lbl_dir)
            if files:
                sample = open(os.path.join(lbl_dir, files[0])).readline().strip()
                parts = sample.split()
                n_coords = len(parts) - 1
                is_seg = n_coords > 4
    total = sum(counts.values())
    fmt = "segmentation" if is_seg else "detection"
    print(f"{name}:")
    print(f"  Classes: {classes}")
    print(f"  Images: {counts} = {total}")
    print(f"  Format: {fmt}")
    print(f"  Sample: {sample[:100]}")
    print()
