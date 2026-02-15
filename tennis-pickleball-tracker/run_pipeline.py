"""Run the full pipeline on Video Project 4.mp4 and produce tracked output."""
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import sys
sys.path.insert(0, "src")

from pipeline import TennisPickleballPipeline, PipelineConfig

cfg = PipelineConfig()
cfg.court_type = "tennis"
cfg.device = "cpu"
cfg.output_codec = "XVID"  # XVID works reliably on Windows

pipeline = TennisPickleballPipeline(cfg)

output_file = "outputs/Video_Project_4_tracked.avi"

print("Processing full video (1139 frames)...")
print("This will take ~5 minutes on CPU...")
stats = pipeline.process_video(
    input_path="data/raw/Video Project 4.mp4",
    output_path=output_file,
    show_preview=False,
)

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
print(f"Total frames: {stats['total_frames']}")
print(f"Processing time: {stats['processing_time']:.1f}s")
print(f"Processing FPS: {stats['processing_fps']:.1f}")
print(f"Ball detections: {stats['ball_detections']}")
print(f"Player detections: {stats['player_detections']}")
bounces = stats.get("bounces", [])
print(f"Bounce events: {len(bounces)}")
for i, b in enumerate(bounces[:10]):
    status = "IN" if b["is_in"] else "OUT"
    print(f"  Bounce {i+1}: frame {b['frame']}, ({b['x']:.2f}, {b['y']:.2f}) - {status}")
if len(bounces) > 10:
    print(f"  ... and {len(bounces) - 10} more bounces")

file_size = os.path.getsize(output_file) / 1024 / 1024
print(f"\nOutput saved to: {output_file} ({file_size:.1f} MB)")
