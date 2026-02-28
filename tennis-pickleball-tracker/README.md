# 🎾 Tennis/Pickleball Detection & Tracking System

A computer vision pipeline for detecting and tracking balls and players in tennis/pickleball videos using a single camera, with 3D trajectory reconstruction and in/out classification.

## ✨ Features

- **Ball Detection & Tracking** — YOLOv8 + TrackNet + Kalman Filter
- **Player Detection & Tracking** — YOLOv8/Faster R-CNN + DeepSORT
- **Court Detection** — Classical CV (HSV + Hough Lines) + CNN keypoint detection
- **3D Trajectory Reconstruction** — Homography + Physics-based parabolic model
- **Bounce Detection** — Physics heuristics + ML classifier (CatBoost)
- **In/Out Classification** — Geometric boundary check with confidence scoring
- **Visualization** — Annotated video output with mini-map, heatmaps, and ball trails

## 🏗️ Architecture

```
Video Input (25-30fps)
        |
   Frame Extraction
        |
   +----+----+
   |         |
Court Det.  Object Det.
(Module 1)  (Module 2)
   |         |
Homography  Ball + Player
   |         |
   +----+----+
        |
  Object Tracking
    (Module 3)
        |
  3D Reconstruction
    (Module 4)
        |
  +-----+-----+
  |     |     |
Bounce  In/  Visual-
Detect  Out  ization
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/AnHgPham/cv.git
cd cv/tennis-pickleball-tracker

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 📂 Dataset

This project uses datasets from [Roboflow](https://roboflow.com/):

| Dataset | Source | Description |
|---------|--------|-------------|
| `pickleball_vision_v1` | Liberin Technologies | Court + Ball detection (YOLOv8 format) |
| `pickleball_court_seg` | Gideons | Court segmentation (~120 images) |
| `pickleball_court_only` | Derived from v1 | Court-only detection for training |

### Download Datasets

```bash
# Set your Roboflow API key
set ROBOFLOW_API_KEY=your_api_key_here    # Windows
# export ROBOFLOW_API_KEY=your_api_key    # Linux/Mac

# Run download script
python download_datasets.py
```

Datasets will be saved to `data/` directory (excluded from git).

## 🚀 Usage

### Quick Start

```bash
python run_pipeline.py input_video.mp4 -o output_video.mp4
```

### Command Line Options

```bash
python -m src.pipeline <input_video> [options]

Options:
  -o, --output        Output video path
  -c, --config        Path to config YAML
  --court-type        tennis | pickleball (default: tennis)
  --detection         yolo | tracknet | classical | combined (default: combined)
  --show              Show preview window
  --max-frames N      Process only first N frames
```

### Python API

```python
from src.pipeline import TennisPickleballPipeline, PipelineConfig

config = PipelineConfig()
config.court_type = "tennis"
config.detection_method = "combined"

pipeline = TennisPickleballPipeline(config)
stats = pipeline.process_video("match.mp4", "match_tracked.mp4")

print(f"Bounces detected: {len(stats['bounces'])}")
for b in stats['bounces']:
    print(f"  Frame {b['frame']}: {'IN' if b['is_in'] else 'OUT'}")
```

### Train Models

```bash
# Train court detector
python train_court_detector.py

# Train YOLOv8 (via Python API)
python -c "
from src.object_detection import YOLODetector
detector = YOLODetector(model_path='yolov8s.pt')
detector.train(data_yaml='configs/yolo_config.yaml', epochs=100, imgsz=640, batch=16)
"
```

## 📁 Project Structure

```
tennis-pickleball-tracker/
├── src/                        # Source code
│   ├── __init__.py
│   ├── court_detection.py      # Module 1: Court detection + Homography
│   ├── object_detection.py     # Module 2: Ball + Player detection
│   ├── object_tracking.py      # Module 3: Kalman, OptFlow, DeepSORT
│   ├── trajectory_3d.py        # Module 4: 3D trajectory reconstruction
│   ├── in_out_classifier.py    # In/Out decision system
│   ├── visualization.py        # Drawing, mini-map, heatmaps
│   └── pipeline.py             # End-to-end pipeline
├── configs/                    # YAML configuration files
│   ├── court_config.yaml
│   ├── tracking_config.yaml
│   ├── tracknet_config.yaml
│   └── yolo_config.yaml
├── data/                       # Datasets (not tracked by git)
├── models/                     # Trained weights (not tracked by git)
├── outputs/                    # Output videos & reports (not tracked by git)
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Unit tests
├── download_datasets.py        # Dataset download script
├── train_court_detector.py     # Court detector training script
├── run_pipeline.py             # Main pipeline runner
├── requirements.txt
└── README.md
```

## 📊 Evaluation Metrics

| Module | Metric | Description |
|--------|--------|-------------|
| Court Detection | Pixel Error | Mean Euclidean error for keypoints |
| Ball Detection | P/R/F1 | Threshold < 5px (TrackNet standard) |
| Player Detection | mAP@0.5 | COCO-style mean average precision |
| Ball Tracking | MOTA/MOTP | Multi-object tracking accuracy/precision |
| Bounce Detection | P/R | ±1 frame tolerance |
| In/Out | Accuracy | Binary classification accuracy |

## 📚 References

1. Huang et al. (2019) — TrackNet: Tracking High-speed and Tiny Objects
2. Redmon et al. (2016) — YOLO: Unified Real-Time Object Detection
3. Bewley et al. (2016) — Simple Online and Realtime Tracking (SORT)
4. Wojke et al. (2017) — DeepSORT: Deep Association Metric
5. Viola & Jones (2001) — Rapid Object Detection using Boosted Cascade
6. Dalal & Triggs (2005) — HOG for Human Detection

## 📄 License

This project is for educational purposes as part of a Computer Vision course.
