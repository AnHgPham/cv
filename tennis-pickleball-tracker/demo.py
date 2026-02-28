"""
Comprehensive Demo Script
=========================
Tests every module and produces annotated output images + video.
"""

import os
import sys
import cv2
import numpy as np
import time
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.court_detection import (
    ClassicalCourtDetector,
    transform_point,
    transform_points,
    draw_court_overlay,
    TENNIS_COURT_CORNERS,
    TENNIS_COURT_KEYPOINTS,
    load_court_config,
)
from src.object_detection import (
    YOLODetector,
    ClassicalBallDetector,
    Detection,
    non_max_suppression,
)
from src.object_tracking import BallTracker, DeepSORTTracker
from src.trajectory_3d import TrajectoryReconstructor
from src.in_out_classifier import EnhancedInOutSystem
from src.visualization import (
    FrameAnnotator,
    MiniMap,
    HeatmapGenerator,
    CompositeFrameBuilder,
)

VIDEO_PATH = "data/raw/Video Project 4.mp4"
OUTPUT_DIR = "outputs/demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAMES_TO_TEST = [0, 30, 60, 74, 100, 150, 200]


def load_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"FAIL: Cannot open video {VIDEO_PATH}")
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h}, {fps:.1f} fps, {total} frames")
    return cap, total, fps, w, h


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


# =========================================================================
# TEST 1: Court Detection & Homography
# =========================================================================
def test_court_detection(cap):
    print("\n" + "=" * 60)
    print("TEST 1: Court Detection & Homography")
    print("=" * 60)

    config = load_court_config("configs/court_config.yaml")
    detector = ClassicalCourtDetector(config)

    results = {}
    for fidx in FRAMES_TO_TEST:
        frame = read_frame(cap, fidx)
        if frame is None:
            print(f"  Frame {fidx}: could not read")
            continue

        H, det_result = detector.detect_and_compute_homography(frame)
        n_h = len(det_result.get("horizontal", []))
        n_v = len(det_result.get("vertical", []))
        n_int = len(det_result.get("intersections", []))
        status = "OK" if H is not None else "FAIL (H=None)"
        print(f"  Frame {fidx:4d}: {status}  |  H={n_h} V={n_v} intersections={n_int}")

        if H is not None:
            results[fidx] = (H, det_result, frame)

            vis = frame.copy()
            # Draw detected lines
            for line in det_result.get("horizontal", []):
                cv2.line(vis, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
            for line in det_result.get("vertical", []):
                cv2.line(vis, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)
            # Draw intersections
            for pt in det_result.get("intersections", []):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
            # Draw selected corners
            corners = det_result.get("selected_corners")
            if corners is not None:
                labels = ["TL", "TR", "BL", "BR"]
                for i, c in enumerate(corners[:4]):
                    cv2.circle(vis, (int(c[0]), int(c[1])), 10, (0, 255, 255), 3)
                    cv2.putText(vis, labels[i], (int(c[0]) + 12, int(c[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw court overlay via inverse H
            try:
                H_inv = np.linalg.inv(H)
                vis = draw_court_overlay(vis, H_inv, TENNIS_COURT_CORNERS, (255, 0, 255), 2)
            except Exception:
                pass

            path = os.path.join(OUTPUT_DIR, f"court_frame{fidx}.jpg")
            cv2.imwrite(path, vis)

    if not results:
        print("  WARNING: No frame produced a valid homography!")
    else:
        print(f"  Valid homographies: {len(results)}/{len(FRAMES_TO_TEST)} frames")
        # Reprojection test on first valid frame
        fidx, (H, _, frame) = next(iter(results.items()))
        H_inv = np.linalg.inv(H)
        court_pts = TENNIS_COURT_CORNERS.reshape(-1, 1, 2).astype(np.float32)
        img_pts = cv2.perspectiveTransform(court_pts, H_inv).reshape(-1, 2)
        reproj = cv2.perspectiveTransform(
            img_pts.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        err = np.mean(np.linalg.norm(reproj - TENNIS_COURT_CORNERS, axis=1))
        print(f"  Reprojection error (frame {fidx}): {err:.4f} m")

    return results


# =========================================================================
# TEST 2: Object Detection (Ball + Player)
# =========================================================================
def test_object_detection(cap):
    print("\n" + "=" * 60)
    print("TEST 2: Object Detection (YOLO + Classical)")
    print("=" * 60)

    yolo = None
    try:
        yolo = YOLODetector(model_path="yolov8s.pt", conf_threshold=0.25, device="cpu")
        print("  YOLO detector loaded OK")
    except Exception as e:
        print(f"  YOLO load failed: {e}")

    classical = ClassicalBallDetector()
    print("  Classical ball detector loaded OK")

    results = {}
    for fidx in FRAMES_TO_TEST:
        frame = read_frame(cap, fidx)
        if frame is None:
            continue

        balls = []
        players = []

        if yolo:
            dets = yolo.detect(frame)
            balls = [d for d in dets if d.class_name in ("ball", "sports ball")]
            players = [d for d in dets if d.class_name in ("player", "person")]
            players = non_max_suppression(players) if players else []

        c_dets = classical.detect(frame)
        c_balls = [d for d in c_dets if d.class_name == "ball"]

        total_balls = len(balls) + len(c_balls)
        print(f"  Frame {fidx:4d}: YOLO balls={len(balls)} players={len(players)}  |  Classical balls={len(c_balls)}")

        results[fidx] = {
            "balls_yolo": balls,
            "balls_classical": c_balls,
            "players": players,
        }

        # Visualize
        vis = frame.copy()
        for det in players:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"person {det.confidence:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for det in balls:
            cx, cy = int(det.center[0]), int(det.center[1])
            cv2.circle(vis, (cx, cy), 8, (0, 255, 255), 2)
            cv2.putText(vis, f"ball {det.confidence:.2f}",
                        (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for det in c_balls:
            cx, cy = int(det.center[0]), int(det.center[1])
            cv2.circle(vis, (cx, cy), 6, (255, 0, 255), 2)

        path = os.path.join(OUTPUT_DIR, f"detect_frame{fidx}.jpg")
        cv2.imwrite(path, vis)

    return results


# =========================================================================
# TEST 3: Object Tracking
# =========================================================================
def test_tracking(cap, fps):
    print("\n" + "=" * 60)
    print("TEST 3: Object Tracking (Kalman + DeepSORT)")
    print("=" * 60)

    yolo = None
    try:
        yolo = YOLODetector(model_path="yolov8s.pt", conf_threshold=0.25, device="cpu")
    except Exception:
        print("  Skipping tracking test: YOLO not available")
        return {}

    ball_tracker = BallTracker()
    player_tracker = DeepSORTTracker()

    n_frames = min(100, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    ball_positions = []
    player_track_counts = []

    for fidx in range(n_frames):
        frame = read_frame(cap, fidx)
        if frame is None:
            break

        dets = yolo.detect(frame)
        ball_dets = [d for d in dets if d.class_name in ("ball", "sports ball")]
        player_dets = [d for d in dets if d.class_name in ("player", "person")]
        player_dets = non_max_suppression(player_dets) if player_dets else []

        ball_det = max(ball_dets, key=lambda d: d.confidence) if ball_dets else None
        ball_pos, _ = ball_tracker.update(frame, ball_det)
        if ball_pos:
            ball_positions.append(ball_pos)

        tracked = player_tracker.update(player_dets, frame)
        player_track_counts.append(len(tracked))

    print(f"  Processed {n_frames} frames")
    print(f"  Ball tracked in {len(ball_positions)}/{n_frames} frames")
    avg_players = np.mean(player_track_counts) if player_track_counts else 0
    print(f"  Avg players tracked per frame: {avg_players:.1f}")

    return {"ball_positions": ball_positions, "n_frames": n_frames}


# =========================================================================
# TEST 4: In/Out Classification
# =========================================================================
def test_in_out(court_results):
    print("\n" + "=" * 60)
    print("TEST 4: In/Out Classification")
    print("=" * 60)

    from src.trajectory_3d import BounceEvent

    system = EnhancedInOutSystem(court_type="tennis_singles")
    classifier = system.in_out_classifier

    if not court_results:
        print("  Skipped: no valid homography from court detection")
        return

    fidx, (H, det_result, frame) = next(iter(court_results.items()))

    test_points_court = [
        (5.0, 5.0, "center court"),
        (11.885, 5.485, "net center"),
        (0.0, 0.0, "top-left corner (on line)"),
        (12.0, 5.0, "just past net"),
        (-1.0, 5.0, "outside top"),
        (25.0, 5.0, "outside bottom"),
        (12.0, -1.0, "outside left"),
        (12.0, 12.0, "outside right"),
        (6.0, 4.0, "service box"),
        (23.77, 10.97, "bottom-right corner"),
    ]

    print(f"  Using homography from frame {fidx}")
    print(f"  Court type: tennis_singles")
    print(f"  {'Point':<30s} {'Court (x,y)':<20s} {'Result':<10s}")
    print(f"  {'-'*60}")

    for cx, cy, label in test_points_court:
        bounce = BounceEvent(frame_id=0, x=cx, y=cy, is_in=False, confidence=1.0)
        is_in, conf = classifier.classify_with_confidence(bounce)
        status = "IN" if is_in else "OUT"
        print(f"  {label:<30s} ({cx:6.2f}, {cy:5.2f})   {status:<5s} conf={conf:.2f}")

    # Test player foot projection
    print(f"\n  Player foot projection test (frame {fidx}):")
    test_pixels = [
        (500, 800, "left side player"),
        (1000, 400, "center far player"),
        (1500, 700, "right side player"),
    ]
    h, w = frame.shape[:2]
    for px, py, label in test_pixels:
        if px < w and py < h:
            court_pos = transform_point(H, np.array([px, py]))
            bounce = BounceEvent(
                frame_id=0, x=float(court_pos[0]), y=float(court_pos[1]),
                is_in=False, confidence=1.0,
            )
            is_in = classifier.classify(bounce)
            status = "IN" if is_in else "OUT"
            print(f"  pixel ({px},{py}) -> court ({court_pos[0]:.2f}, {court_pos[1]:.2f}) = {status}  [{label}]")


# =========================================================================
# TEST 5: Visualization & Mini-map
# =========================================================================
def test_visualization(cap, court_results):
    print("\n" + "=" * 60)
    print("TEST 5: Visualization & Mini-map")
    print("=" * 60)

    annotator = FrameAnnotator()
    minimap = MiniMap(court_type="tennis")
    composite = CompositeFrameBuilder()

    if not court_results:
        print("  Limited test: no homography available")
        fidx = 74
        frame = read_frame(cap, fidx)
        if frame is None:
            print("  Could not read frame")
            return
        minimap_img = minimap.render()
        output = composite.build(frame, minimap=minimap_img, info={"Frame": str(fidx)})
        path = os.path.join(OUTPUT_DIR, f"vis_frame{fidx}.jpg")
        cv2.imwrite(path, output)
        print(f"  Saved: {path}")
        return

    for fidx, (H, det_result, frame) in court_results.items():
        # Simulate ball and player positions on minimap
        ball_court = (11.0, 5.0)
        player_positions = [(3.0, 3.0), (3.0, 8.0), (20.0, 3.0), (20.0, 8.0)]

        minimap_img = minimap.render(
            ball_court_pos=ball_court,
            player_court_positions=player_positions,
        )

        info = {
            "Frame": str(fidx),
            "Ball": "Simulated",
            "Players": "4",
            "Court": "Detected",
        }

        annotated = frame.copy()
        try:
            H_inv = np.linalg.inv(H)
            annotated = draw_court_overlay(annotated, H_inv, TENNIS_COURT_CORNERS, (0, 255, 0), 2)
        except Exception:
            pass
        annotated = annotator.draw_info_overlay(annotated, info)

        output = composite.build(annotated, minimap=minimap_img, info=info)
        path = os.path.join(OUTPUT_DIR, f"vis_frame{fidx}.jpg")
        cv2.imwrite(path, output)
        print(f"  Saved composite frame: {path}")
        break  # just first valid frame for this test


# =========================================================================
# TEST 6: Full Pipeline (short clip)
# =========================================================================
def test_full_pipeline(cap, fps, total_frames):
    print("\n" + "=" * 60)
    print("TEST 6: Full Pipeline - Annotated Video Demo")
    print("=" * 60)

    from src.pipeline import TennisPickleballPipeline, PipelineConfig

    config = PipelineConfig()
    config.court_type = "tennis"
    config.detection_method = "combined"
    config.player_method = "yolo"
    config.show_minimap = True
    config.show_trajectory = True
    config.show_heatmap = False
    config.output_codec = "MJPG"
    config.device = "cpu"

    pipeline = TennisPickleballPipeline(config)

    n_frames = min(200, total_frames)
    out_path = os.path.join(OUTPUT_DIR, "demo_output.avi")

    writer = None
    start = time.time()
    valid_court_frames = 0
    ball_detected_frames = 0
    player_frames = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for fidx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame, fidx)
        output_frame = pipeline._build_output_frame(frame, result)

        if writer is None:
            h_out, w_out = output_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w_out, h_out))
            print(f"  Output resolution: {w_out}x{h_out}")

        writer.write(output_frame)

        if result.get("court_detected"):
            valid_court_frames += 1
        if result.get("ball_detected"):
            ball_detected_frames += 1
        if result.get("player_tracks"):
            player_frames += 1

        # Save key frames as images
        if fidx in [0, 30, 74, 100, 150, 199]:
            img_path = os.path.join(OUTPUT_DIR, f"pipeline_frame{fidx}.jpg")
            cv2.imwrite(img_path, output_frame)

    elapsed = time.time() - start
    if writer:
        writer.release()

    print(f"  Processed {n_frames} frames in {elapsed:.1f}s ({n_frames/elapsed:.1f} FPS)")
    print(f"  Court detected: {valid_court_frames}/{n_frames} frames")
    print(f"  Ball detected:  {ball_detected_frames}/{n_frames} frames")
    print(f"  Players found:  {player_frames}/{n_frames} frames")
    print(f"  Output video:   {out_path}")

    # Check file size
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"  Output file size: {size / 1024 / 1024:.1f} MB")
        if size < 1000:
            print("  WARNING: Output file is very small - may be corrupt!")
    else:
        print("  WARNING: Output file was not created!")

    return {
        "court_frames": valid_court_frames,
        "ball_frames": ball_detected_frames,
        "player_frames": player_frames,
        "total": n_frames,
    }


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 60)
    print("Tennis/Pickleball Detection & Tracking - Full Demo")
    print("=" * 60)

    cap, total, fps, w, h = load_video()

    # Run tests
    test_results = {}

    try:
        court_results = test_court_detection(cap)
        test_results["court"] = "PASS" if court_results else "FAIL"
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        court_results = {}
        test_results["court"] = f"ERROR: {e}"

    try:
        det_results = test_object_detection(cap)
        test_results["detection"] = "PASS"
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        test_results["detection"] = f"ERROR: {e}"

    try:
        track_results = test_tracking(cap, fps)
        test_results["tracking"] = "PASS"
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        test_results["tracking"] = f"ERROR: {e}"

    try:
        test_in_out(court_results)
        test_results["in_out"] = "PASS"
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        test_results["in_out"] = f"ERROR: {e}"

    try:
        test_visualization(cap, court_results)
        test_results["visualization"] = "PASS"
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        test_results["visualization"] = f"ERROR: {e}"

    try:
        pipeline_stats = test_full_pipeline(cap, fps, total)
        test_results["pipeline"] = "PASS"
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        test_results["pipeline"] = f"ERROR: {e}"
        pipeline_stats = {}

    cap.release()

    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    for module, status in test_results.items():
        icon = "[OK]" if status == "PASS" else "[!!]"
        print(f"  {icon} {module}: {status}")

    if pipeline_stats:
        print(f"\n  Pipeline output: {OUTPUT_DIR}/demo_output.avi")
        print(f"  Key frame images: {OUTPUT_DIR}/pipeline_frame*.jpg")

    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
