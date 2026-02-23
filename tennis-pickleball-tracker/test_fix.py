"""Test fixed pipeline: player filtering + ball detection."""
import sys, os, cv2
sys.path.insert(0, "src")
os.chdir(r"D:\Downloads\cv\tennis-pickleball-tracker")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from pipeline import TennisPickleballPipeline, PipelineConfig

cfg = PipelineConfig()
cfg.court_type = "tennis"
cfg.device = "cpu"
pipeline = TennisPickleballPipeline(cfg)

cap = cv2.VideoCapture("data/raw/Video Project 4.mp4")

# Process 80 frames (ball visible around frame 57-80)
for frame_idx in range(80):
    ret, frame = cap.read()
    if not ret:
        break
    result = pipeline.process_frame(frame, frame_idx)

    n = len(result["player_tracks"])
    ball = result["ball_detected"]
    ball_pos = result.get("ball_position")

    if frame_idx % 10 == 0 or frame_idx >= 70:
        players = ", ".join(
            f"P{tid}({det.bbox[0]},{det.bbox[1]}-{det.bbox[2]},{det.bbox[3]})"
            for tid, det in result["player_tracks"]
        )
        poly = "yes" if pipeline.court_image_polygon is not None else "no"
        print(
            f"F{frame_idx:3d}: poly={poly} players={n} ball={ball} "
            f"pos={ball_pos} [{players}]"
        )

# Save annotated frame at frame 74
cap.set(cv2.CAP_PROP_POS_FRAMES, 74)
ret, frame74 = cap.read()
cap.release()

# Re-process frame 74 to get fresh result (pipeline state is at frame 79)
# Just build output from last processed frame
result = pipeline.process_frame(frame74, 74)
output = pipeline._build_output_frame(frame74, result)

cv2.imwrite("outputs/test_results/annotated_frame_v2.jpg", output)
print("\nSaved outputs/test_results/annotated_frame_v2.jpg")
print(f"Final: players={len(result['player_tracks'])}, ball={result['ball_detected']}")
