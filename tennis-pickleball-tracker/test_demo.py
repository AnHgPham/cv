"""Quick demo: test all pipeline modules with synthetic frames."""
import os, sys
sys.path.insert(0, r"D:\Downloads\cv\tennis-pickleball-tracker\src")
os.chdir(r"D:\Downloads\cv\tennis-pickleball-tracker")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import cv2
import numpy as np
import time

def create_synthetic_court_frame(w=1280, h=720):
    """Create a synthetic frame with a pickleball court-like shape."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Blue court surface
    court_pts = np.array([
        [400, 150], [880, 150],  # top
        [200, 600], [1080, 600],  # bottom
    ], dtype=np.int32)
    cv2.fillConvexPoly(frame, court_pts, (180, 120, 40))  # blue court

    # White court lines
    cv2.line(frame, (400, 150), (880, 150), (255, 255, 255), 2)
    cv2.line(frame, (200, 600), (1080, 600), (255, 255, 255), 2)
    cv2.line(frame, (400, 150), (200, 600), (255, 255, 255), 2)
    cv2.line(frame, (880, 150), (1080, 600), (255, 255, 255), 2)
    # Center line
    cv2.line(frame, (640, 150), (640, 600), (255, 255, 255), 2)
    # Kitchen lines
    cv2.line(frame, (480, 280), (800, 280), (255, 255, 255), 2)
    cv2.line(frame, (350, 470), (930, 470), (255, 255, 255), 2)

    # Fake ball
    ball_x = np.random.randint(400, 800)
    ball_y = np.random.randint(200, 500)
    cv2.circle(frame, (ball_x, ball_y), 6, (0, 255, 255), -1)

    # Fake players (rectangles)
    cv2.rectangle(frame, (300, 400), (360, 560), (0, 200, 0), -1)
    cv2.rectangle(frame, (900, 400), (960, 560), (0, 0, 200), -1)

    return frame

print("=" * 60)
print("DEMO: Tennis/Pickleball Pipeline Module Test")
print("=" * 60)

# ── Test 1: Module imports ──
print("\n[1/6] Testing module imports...")
try:
    from pipeline import TennisPickleballPipeline, PipelineConfig
    from court_detection import (
        ClassicalCourtDetector, SegmentationCourtDetector,
        extract_court_corners_from_segmentation, 
        PICKLEBALL_COURT_CORNERS, TENNIS_COURT_CORNERS,
    )
    from object_detection import YOLODetector, Detection
    from object_tracking import BallTracker, DeepSORTTracker
    from visualization import FrameAnnotator, MiniMap, CompositeFrameBuilder
    print("  ✓ All modules imported successfully")
except Exception as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# ── Test 2: SegmentationCourtDetector ──
print("\n[2/6] Testing SegmentationCourtDetector...")
seg_model = "models/pickleball_court/best.pt"
if os.path.exists(seg_model):
    print(f"  ✓ Seg model found: {seg_model}")
    try:
        seg_det = SegmentationCourtDetector(model_path=seg_model, conf_threshold=0.3)
        frame = create_synthetic_court_frame()
        H, result = seg_det.detect_and_compute_homography(frame)
        print(f"  ✓ SegmentationCourtDetector initialized and ran")
        print(f"    Homography found: {H is not None}")
        print(f"    Result keys: {list(result.keys())}")
    except Exception as e:
        print(f"  ⚠ SegmentationCourtDetector error (expected with synthetic): {e}")
else:
    print(f"  ✗ Seg model not found at {seg_model}")

# ── Test 3: ClassicalCourtDetector ──
print("\n[3/6] Testing ClassicalCourtDetector...")
try:
    from court_detection import load_court_config
    court_cfg = load_court_config("configs/court_config.yaml")
    classical = ClassicalCourtDetector(court_cfg)
    frame = create_synthetic_court_frame()
    result = classical.detect(frame)
    print(f"  ✓ ClassicalCourtDetector ran successfully")
    print(f"    Lines found: {len(result.get('lines', []))}")
    print(f"    Intersections: {len(result.get('intersections', []))}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# ── Test 4: Pipeline init (pickleball + seg) ──
print("\n[4/6] Testing pipeline init with pickleball + seg model...")
try:
    cfg = PipelineConfig()
    cfg.court_type = "pickleball"
    cfg.device = "cpu"
    cfg.seg_model_path = seg_model
    pipeline = TennisPickleballPipeline(cfg)
    detector_type = type(pipeline.court_detector).__name__
    print(f"  ✓ Pipeline initialized")
    print(f"    Court detector: {detector_type}")
    print(f"    Court corners shape: {pipeline.court_corners.shape}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback; traceback.print_exc()

# ── Test 5: Process synthetic frames ──
print("\n[5/6] Processing 10 synthetic frames...")
try:
    start = time.time()
    for i in range(10):
        frame = create_synthetic_court_frame()
        result = pipeline.process_frame(frame, i)
    elapsed = time.time() - start
    print(f"  ✓ Processed 10 frames in {elapsed:.2f}s ({10/elapsed:.1f} FPS)")
    print(f"    Court detected: {result.get('court_detected')}")
    print(f"    Ball detected: {result.get('ball_detected')}")
    print(f"    Players tracked: {len(result.get('player_tracks', []))}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback; traceback.print_exc()

# ── Test 6: Build output frame ──
print("\n[6/6] Testing output frame generation...")
try:
    frame = create_synthetic_court_frame()
    result = pipeline.process_frame(frame, 11)
    output = pipeline._build_output_frame(frame, result)
    print(f"  ✓ Output frame: {output.shape[1]}x{output.shape[0]}")

    os.makedirs("outputs/demo", exist_ok=True)
    cv2.imwrite("outputs/demo/synthetic_test_frame.jpg", output)
    print(f"  ✓ Saved: outputs/demo/synthetic_test_frame.jpg")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("DEMO COMPLETE")
print("=" * 60)
