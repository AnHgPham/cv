"""Run pipeline demo on TrackNet video."""
import os, sys
sys.path.insert(0, r"D:\Downloads\cv\tennis-pickleball-tracker\src")
os.chdir(r"D:\Downloads\cv\tennis-pickleball-tracker")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from pipeline import TennisPickleballPipeline, PipelineConfig
import cv2
import time

cfg = PipelineConfig()
cfg.court_type = "tennis"
cfg.device = "cpu"

print("Initializing pipeline...")
pipeline = TennisPickleballPipeline(cfg)

INPUT = "data/raw/tracknet_tennis.mp4"
OUTPUT = "outputs/tracknet_demo_output.avi"
MAX_FRAMES = 100

# Open input
cap = cv2.VideoCapture(INPUT)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_FRAMES)

print(f"Input: {width}x{height} @ {fps}fps, processing {total} frames")

# Process first frame to get output size
ret, first_frame = cap.read()
result = pipeline.process_frame(first_frame, 0)
output_frame = pipeline._build_output_frame(first_frame, result)
out_h, out_w = output_frame.shape[:2]

# Setup writer
os.makedirs("outputs", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(OUTPUT, fourcc, fps, (out_w, out_h))
writer.write(output_frame)

ball_dets = 1 if result.get("ball_detected") else 0
player_dets = len(result.get("player_tracks", []))

start = time.time()
from tqdm import tqdm
progress = tqdm(total=total, initial=1, desc="Processing")

for i in range(1, total):
    ret, frame = cap.read()
    if not ret:
        break
    result = pipeline.process_frame(frame, i)
    out = pipeline._build_output_frame(frame, result)
    writer.write(out)
    if result.get("ball_detected"):
        ball_dets += 1
    player_dets += len(result.get("player_tracks", []))
    progress.update(1)

progress.close()
cap.release()
writer.release()

elapsed = time.time() - start
file_size = os.path.getsize(OUTPUT) / 1024 / 1024

print()
print("=" * 50)
print("PIPELINE RESULTS")
print("=" * 50)
print(f"Output: {OUTPUT} ({file_size:.1f} MB)")
print(f"Output size: {out_w}x{out_h}")
print(f"Frames: {total}")
print(f"Time: {elapsed:.1f}s")
print(f"FPS: {total / elapsed:.1f}")
print(f"Ball detections: {ball_dets}")
print(f"Player detections: {player_dets}")

# Save a sample frame
sample = cv2.VideoCapture(OUTPUT)
sample.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, s_frame = sample.read()
if ret:
    cv2.imwrite("outputs/demo/tracknet_sample_frame.jpg", s_frame)
    print("Sample frame saved: outputs/demo/tracknet_sample_frame.jpg")
sample.release()
