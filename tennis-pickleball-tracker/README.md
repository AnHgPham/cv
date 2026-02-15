# Tennis/Pickleball Detection & Tracking System

A computer vision pipeline for detecting and tracking balls and players in tennis/pickleball videos using a single camera, with 3D trajectory reconstruction and in/out classification.

## Project Overview

This system processes video from a single camera (25-30 fps) to:

- **Detect and track** ball trajectory in real-time
- **Detect and track** player positions with ID assignment
- **Detect court lines** and compute homography for coordinate mapping
- **Reconstruct 3D trajectory** of the ball using physics models
- **Classify in/out** for bounce events
- **Generate heatmaps** for ball landing zones and player movement
- **Visualize** results with annotated video and mini-map

## Architecture

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

## Modules

### Module 1: Court Detection & Homography
- **Classical**: HSV thresholding -> Morphology -> Canny -> Hough Lines -> RANSAC Homography
- **Deep Learning**: CNN keypoint detection (ResNet-18 backbone) -> Homography
- **SIFT Matching**: For tracking court between frames with camera motion

### Module 2: Object Detection
- **Viola-Jones + HOG + SVM** (Feature Engineering baseline)
- **YOLOv8** (Feature Learning - fine-tuned on tennis/pickleball)
- **TrackNet** (Specialized heatmap network for small ball detection using 3 consecutive frames)
- **Faster R-CNN** (Two-stage detector for player detection)

### Module 3: Object Tracking
- **Kalman Filter**: Linear state estimation for ball tracking `[x, y, vx, vy]`
- **Optical Flow (Lucas-Kanade)**: Motion estimation to supplement Kalman
- **DeepSORT**: Multi-object tracking with deep appearance features for players

### Module 4: 3D Trajectory Reconstruction
- **Court Projection**: Homography-based 2D -> court coordinate mapping
- **Physics Model**: Parabolic trajectory under gravity for height estimation
- **Extended Kalman Filter**: 6D state `[x, y, z, vx, vy, vz]`
- **Bounce Detection**: Physics heuristics + ML classifier (RandomForest/CatBoost)
- **In/Out Classification**: Geometric boundary check with confidence

## Installation

```bash
# Clone or download the project
cd tennis-pickleball-tracker

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for YOLO and TrackNet)

## Usage

### Quick Start - Process a Video
```bash
python -m src.pipeline input_video.mp4 -o output_video.mp4 --show
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

### Data Preprocessing
```python
from src.pipeline import DataPreprocessor

preprocessor = DataPreprocessor()

# Extract frames from video
preprocessor.extract_frames("match.mp4", "data/frames/raw/")

# Resize for TrackNet (640x360)
preprocessor.resize_frames("data/frames/raw/", "data/frames/tracknet/", (640, 360))

# Resize for YOLO (640x640)
preprocessor.resize_frames("data/frames/raw/", "data/frames/yolo/", (640, 640))

# Split dataset 70/15/15
DataPreprocessor.split_dataset(
    "data/frames/raw/", "data/labels/",
    "data/", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
)
```

### Train YOLOv8
```python
from src.object_detection import YOLODetector

detector = YOLODetector(model_path="yolov8s.pt")
detector.train(
    data_yaml="configs/yolo_data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)
```

## Project Structure

```
tennis-pickleball-tracker/
├── data/
│   ├── raw/                    # Raw video files
│   ├── frames/                 # Extracted frames
│   ├── labels/                 # Annotation files
│   └── augmented/              # Augmented data
├── models/
│   ├── court_detector/         # Court detection weights
│   ├── ball_detector/          # YOLO + TrackNet weights
│   ├── player_detector/        # Faster R-CNN weights
│   └── bounce_classifier/      # Bounce ML model
├── src/
│   ├── __init__.py
│   ├── court_detection.py      # Module 1: Court + Homography
│   ├── object_detection.py     # Module 2: Ball + Player detection
│   ├── object_tracking.py      # Module 3: Kalman, OptFlow, DeepSORT
│   ├── trajectory_3d.py        # Module 4: 3D reconstruction
│   ├── in_out_classifier.py    # In/Out decision system
│   ├── visualization.py        # Drawing, mini-map, heatmaps
│   └── pipeline.py             # End-to-end pipeline
├── notebooks/                  # Jupyter notebooks
├── configs/                    # YAML configuration files
├── outputs/                    # Output videos and reports
├── requirements.txt
└── README.md
```

## Evaluation Metrics

| Module | Metric | Description |
|--------|--------|-------------|
| Court Detection | Pixel Error | Mean Euclidean error for keypoints |
| Ball Detection | P/R/F1 | Threshold < 5 pixels (TrackNet standard) |
| Player Detection | mAP@0.5 | COCO-style mean average precision |
| Ball Tracking | MOTA/MOTP | Multi-object tracking accuracy/precision |
| Bounce Detection | P/R | +/- 1 frame tolerance |
| In/Out | Accuracy | Binary classification accuracy |

## References

1. Huang et al. (2019) - TrackNet: Tracking High-speed and Tiny Objects
2. Redmon et al. (2016) - YOLO: Unified Real-Time Object Detection
3. Bewley et al. (2016) - Simple Online and Realtime Tracking (SORT)
4. Wojke et al. (2017) - DeepSORT: Deep Association Metric
5. Viola & Jones (2001) - Rapid Object Detection using Boosted Cascade
6. Dalal & Triggs (2005) - HOG for Human Detection

## License

This project is for educational purposes as part of a Computer Vision course.
