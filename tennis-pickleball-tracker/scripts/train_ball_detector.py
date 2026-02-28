"""
Train YOLOv8 Ball Detection for Pickleball
==========================================

Hướng dẫn sử dụng trên Google Colab Pro:

1. Upload thư mục data/merged_ball/ lên Google Drive
2. Copy script này lên Colab
3. Chạy từng cell

Hoặc chạy local nếu có GPU:
    python scripts/train_ball_detector.py
"""

import os
import sys

# === CONFIG ===
# Đổi path nếu chạy trên Colab
# COLAB: DATA_PATH = "/content/drive/MyDrive/pickleball/merged_ball"
# LOCAL: DATA_PATH = "data/merged_ball"
DATA_PATH = os.environ.get("BALL_DATA_PATH", "data/merged_ball")
DATA_YAML = os.path.join(DATA_PATH, "data.yaml")

MODEL_BASE = "yolov8n.pt"       # nano (nhanh, ~6MB)
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16                  # Giảm nếu hết VRAM (8→4)
PATIENCE = 20                    # Early stopping
PROJECT = "runs/detect"
NAME = "pickleball_ball"

# === SETUP (Colab) ===
def setup_colab():
    """Run this if on Google Colab."""
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    if IN_COLAB:
        print("🔗 Mounting Google Drive...")
        from google.colab import drive
        drive.mount('/content/drive')
        
        print("📦 Installing ultralytics...")
        os.system("pip install -q ultralytics")
        
        # Check GPU
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"✅ GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            print("⚠️ No GPU detected! Training will be very slow.")
    
    return IN_COLAB


def train():
    """Train YOLOv8 on merged ball dataset."""
    from ultralytics import YOLO
    import yaml
    
    # Verify dataset
    if not os.path.exists(DATA_YAML):
        print(f"❌ data.yaml not found: {DATA_YAML}")
        print(f"   Make sure merged_ball/ is at: {DATA_PATH}")
        sys.exit(1)
    
    with open(DATA_YAML) as f:
        data = yaml.safe_load(f)
    print(f"📊 Dataset: {DATA_PATH}")
    print(f"   Classes: {data.get('names', [])}")
    
    # Count images
    for split in ['train', 'val', 'test']:
        split_key = split
        if split == 'val':
            split_key = 'valid'
        img_dir = os.path.join(DATA_PATH, data.get(split, f'{split_key}/images'))
        if os.path.exists(img_dir):
            n = len(os.listdir(img_dir))
            print(f"   {split}: {n} images")
    
    # Load model
    print(f"\n🏗️ Loading base model: {MODEL_BASE}")
    model = YOLO(MODEL_BASE)
    
    # Train
    print(f"\n🚀 Training: {EPOCHS} epochs, imgsz={IMG_SIZE}, batch={BATCH_SIZE}")
    print(f"   Output: {PROJECT}/{NAME}/")
    print("=" * 50)
    
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        verbose=True,
        # Augmentation
        hsv_h=0.015,      # HSV-Hue augmentation
        hsv_s=0.7,        # HSV-Saturation
        hsv_v=0.4,        # HSV-Value
        degrees=10.0,     # Rotation
        translate=0.1,    # Translation
        scale=0.5,        # Scale
        fliplr=0.5,       # Horizontal flip
        mosaic=1.0,       # Mosaic augmentation
        mixup=0.1,        # MixUp augmentation
    )
    
    # Results
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    
    best_pt = os.path.join(PROJECT, NAME, "weights", "best.pt")
    if os.path.exists(best_pt):
        size_mb = os.path.getsize(best_pt) / 1e6
        print(f"✅ Best model: {best_pt} ({size_mb:.1f} MB)")
        print(f"\n📋 Next steps:")
        print(f"   1. Download best.pt from Colab")
        print(f"   2. Copy to: models/pickleball_ball/best.pt")
        print(f"   3. Run pipeline: python run_match_demo.py")
    else:
        print(f"⚠️ best.pt not found at {best_pt}")
    
    return results


def evaluate():
    """Evaluate trained model."""
    from ultralytics import YOLO
    
    best_pt = os.path.join(PROJECT, NAME, "weights", "best.pt")
    if not os.path.exists(best_pt):
        print(f"❌ Model not found: {best_pt}")
        return
    
    print(f"📊 Evaluating: {best_pt}")
    model = YOLO(best_pt)
    results = model.val(data=DATA_YAML, imgsz=IMG_SIZE)
    
    print(f"\n📈 Results:")
    print(f"   mAP@0.5:      {results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"   Precision:     {results.box.mp:.4f}")
    print(f"   Recall:        {results.box.mr:.4f}")
    
    return results


if __name__ == "__main__":
    in_colab = setup_colab()
    
    if in_colab:
        # On Colab, update DATA_PATH to Drive location
        # Change this to match your Drive structure
        pass
    
    train()
    print("\n")
    evaluate()
