"""Run the full pipeline on the input video."""
import sys, os
sys.path.insert(0, r"D:\Downloads\cv\tennis-pickleball-tracker\src")
os.chdir(r"D:\Downloads\cv\tennis-pickleball-tracker")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from pipeline import TennisPickleballPipeline, PipelineConfig

cfg = PipelineConfig()
cfg.court_type = "tennis"
cfg.device = "cpu"

pipeline = TennisPickleballPipeline(cfg)
stats = pipeline.process_video(
    r"data/raw/Video Project 4.mp4",
    r"outputs/Video_Project_4_tracked.mp4",
    show_preview=False,
)

print("\n=== Done ===")
print(f"Frames: {stats['total_frames']}")
print(f"Time: {stats['processing_time']:.1f}s")
print(f"FPS: {stats['processing_fps']:.1f}")
print(f"Ball detections: {stats['ball_detections']}")
print(f"Player detections: {stats['player_detections']}")

"""Run the full pipeline with MJPG codec (reliable on Windows)."""
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import sys
sys.path.insert(0, "src")
import cv2
import numpy as np
import time

from pipeline import TennisPickleballPipeline, PipelineConfig

cfg = PipelineConfig()
cfg.court_type = "tennis"
cfg.device = "cpu"

pipeline = TennisPickleballPipeline(cfg)

# Open input video
cap = cv2.VideoCapture("data/raw/Video Project 4.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Process first frame to determine actual output size
ret, first_frame = cap.read()
result = pipeline.process_frame(first_frame, 0)
output_frame = pipeline._build_output_frame(first_frame, result)
out_h, out_w = output_frame.shape[:2]

print(f"Input: {width}x{height} @ {fps}fps, {total_frames} frames")
print(f"Output frame: {out_w}x{out_h}")

# Setup writer with MJPG codec (works universally on Windows)
output_path = "outputs/Video_Project_4_tracked.avi"
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

if not writer.isOpened():
    print("ERROR: VideoWriter failed to open!")
    sys.exit(1)

# Write first frame
writer.write(output_frame)

# Process remaining frames
start_time = time.time()
ball_detections = 0
if result.get("ball_detected"):
    ball_detections += 1

from tqdm import tqdm
progress = tqdm(total=total_frames, desc="Processing video", initial=1)

for frame_idx in range(1, total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    result = pipeline.process_frame(frame, frame_idx)
    output_frame = pipeline._build_output_frame(frame, result)
    writer.write(output_frame)
    
    if result.get("ball_detected"):
        ball_detections += 1
    
    progress.update(1)

progress.close()
cap.release()
writer.release()

elapsed = time.time() - start_time
file_size = os.path.getsize(output_path) / 1024 / 1024

# Get trajectory analysis
bounces = []
if pipeline.trajectory_reconstructor:
    trajectory = pipeline.trajectory_reconstructor.finalize()
    analysis = pipeline.in_out_system.analyze_trajectory(trajectory)
    bounces = analysis.get("bounces", [])

# Write results to file for easy reading
with open("outputs/pipeline_results.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("Pipeline Processing Complete!\n")
    f.write("=" * 60 + "\n")
    f.write(f"Input: data/raw/Video Project 4.mp4\n")
    f.write(f"Output: {output_path}\n")
    f.write(f"File size: {file_size:.1f} MB\n")
    f.write(f"Total frames: {total_frames}\n")
    f.write(f"Processing time: {elapsed:.1f}s\n")
    f.write(f"Processing FPS: {total_frames / elapsed:.1f}\n")
    f.write(f"Ball detections: {ball_detections}\n")
    f.write(f"Bounce events: {len(bounces)}\n")
    for i, b in enumerate(bounces):
        status = "IN" if b.is_in else "OUT"
        f.write(f"  Bounce {i+1}: frame {b.frame_id}, "
                f"({b.x:.2f}, {b.y:.2f}) - {status} "
                f"(conf: {b.confidence:.2f})\n")

# Print to console too
with open("outputs/pipeline_results.txt") as f:
    print(f.read())
