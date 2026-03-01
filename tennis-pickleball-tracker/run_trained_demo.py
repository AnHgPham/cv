"""Run pipeline with TRAINED ball detector on pickleball match video."""
import os, sys
sys.path.insert(0, r"D:\Downloads\cv\tennis-pickleball-tracker\src")
os.chdir(r"D:\Downloads\cv\tennis-pickleball-tracker")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from pipeline import TennisPickleballPipeline, PipelineConfig
import cv2
import numpy as np
import time

cfg = PipelineConfig()
cfg.court_type = "pickleball"
cfg.device = "cpu"
cfg.seg_model_path = "models/pickleball_court/best.pt"

# === USE TRAINED BALL MODEL ===
cfg.yolo_model = "models/ball_detector/best.pt"
cfg.config["is_custom_yolo"] = True
cfg.detection_method = "yolo"
cfg.yolo_conf = 0.3

print("Initializing pipeline (pickleball + trained ball model)...")
pipeline = TennisPickleballPipeline(cfg)
print(f"Court detector: {type(pipeline.court_detector).__name__}")
print(f"Ball model: {cfg.yolo_model}")
print(f"Court model: {cfg.seg_model_path}")

INPUT = "data/raw/pickleball_match.mp4"
OUTPUT = "outputs/pickleball_trained_output.avi"
MAX_FRAMES = 200

cap = cv2.VideoCapture(INPUT)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_FRAMES)

print(f"Input: {width}x{height} @ {fps:.0f}fps, processing {total} frames")

ret, first_frame = cap.read()
result = pipeline.process_frame(first_frame, 0)
output_frame = pipeline._build_output_frame(first_frame, result)
out_h, out_w = output_frame.shape[:2]

os.makedirs("outputs/demo", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(OUTPUT, fourcc, fps, (out_w, out_h))
writer.write(output_frame)

ball_dets = 1 if result.get("ball_detected") else 0
court_dets = 1 if result.get("court_detected") else 0

start = time.time()
from tqdm import tqdm
progress = tqdm(total=total, initial=1, desc="Trained Ball Model")

for i in range(1, total):
    ret, frame = cap.read()
    if not ret:
        break
    result = pipeline.process_frame(frame, i)
    out = pipeline._build_output_frame(frame, result)
    writer.write(out)
    if result.get("ball_detected"):
        ball_dets += 1
    if result.get("court_detected"):
        court_dets += 1
    progress.update(1)

progress.close()
cap.release()
writer.release()

elapsed = time.time() - start
file_size = os.path.getsize(OUTPUT) / 1024 / 1024

print()
print("=" * 55)
print("TRAINED BALL MODEL RESULTS")
print("=" * 55)
print(f"Ball model: {cfg.yolo_model}")
print(f"Output: {OUTPUT} ({file_size:.1f} MB)")
print(f"Frames: {total} | Time: {elapsed:.1f}s | FPS: {total/elapsed:.1f}")
print(f"Ball detections: {ball_dets}/{total} ({100*ball_dets/total:.0f}%)")
print(f"Court detections: {court_dets}/{total}")

# Save sample frames
for idx in [0, 50, 100, 150, 199]:
    if idx < total:
        s_cap = cv2.VideoCapture(OUTPUT)
        s_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, sf = s_cap.read()
        if ret:
            path = f"outputs/demo/trained_frame_{idx:04d}.jpg"
            cv2.imwrite(path, sf)
            print(f"Saved: {path}")
        s_cap.release()
