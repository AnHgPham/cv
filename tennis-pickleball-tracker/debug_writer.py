"""Debug VideoWriter issue — test what frame size the pipeline actually outputs."""
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import sys
sys.path.insert(0, "src")
import cv2
import numpy as np

from pipeline import TennisPickleballPipeline, PipelineConfig

cfg = PipelineConfig()
cfg.court_type = "tennis"
cfg.device = "cpu"

pipeline = TennisPickleballPipeline(cfg)

# Read first frame
cap = cv2.VideoCapture("data/raw/Video Project 4.mp4")
ret, frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"Input: {width}x{height} @ {fps} fps")

# Process one frame to see the actual output size
result = pipeline.process_frame(frame, 0)
output_frame = pipeline._build_output_frame(frame, result)
print(f"Output frame shape: {output_frame.shape}")
print(f"Output frame dtype: {output_frame.dtype}")

out_h, out_w = output_frame.shape[:2]
print(f"Output dimensions: {out_w}x{out_h}")
print(f"Expected by writer: {width + 220}x{height}")

# Test VideoWriter with correct dimensions
codecs = ["XVID", "MJPG", "mp4v", "X264", "DIVX"]
for codec in codecs:
    fname = f"outputs/test_codec_{codec}.avi"
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(fname, fourcc, fps, (out_w, out_h))
        if writer.isOpened():
            writer.write(output_frame)
            writer.write(output_frame)
            writer.release()
            size = os.path.getsize(fname)
            print(f"  {codec}: {size} bytes {'OK' if size > 100 else 'EMPTY'}")
        else:
            print(f"  {codec}: FAILED to open writer")
    except Exception as e:
        print(f"  {codec}: ERROR {e}")

# Save a frame as image for reference
cv2.imwrite("outputs/test_output_frame.jpg", output_frame)
print(f"\nSaved test frame: outputs/test_output_frame.jpg ({os.path.getsize('outputs/test_output_frame.jpg') / 1024:.0f} KB)")
