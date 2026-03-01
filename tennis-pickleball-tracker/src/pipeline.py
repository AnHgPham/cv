"""
Pipeline: End-to-End Integration
=================================

Integrates all modules into a complete video processing pipeline:

1. Video Input -> Frame Extraction
2. Court Detection -> Homography (Module 1)
3. Object Detection -> Ball + Player positions (Module 2)
4. Object Tracking -> Smoothed trajectories (Module 3)
5. 3D Reconstruction -> Bounce detection + In/Out (Module 4)
6. Visualization -> Annotated output video

Usage:
    pipeline = TennisPickleballPipeline(config)
    pipeline.process_video("input.mp4", "output.mp4")
"""

import cv2
import numpy as np
import yaml
import os
import time
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm

# Module imports
try:
    from .court_detection import (
        ClassicalCourtDetector,
        DeepCourtDetector,
        SegmentationCourtDetector,
        SIFTCourtMatcher,
        transform_point,
        TENNIS_COURT_CORNERS,
        PICKLEBALL_COURT_CORNERS,
    )
    from .object_detection import (
        Detection,
        YOLODetector,
        TrackNetDetector,
        ClassicalPlayerDetector,
        ClassicalBallDetector,
        FasterRCNNDetector,
        non_max_suppression,
    )
    from .object_tracking import (
        BallTracker,
        DeepSORTTracker,
        Track,
    )
    from .trajectory_3d import (
        TrajectoryReconstructor,
        Trajectory3D,
    )
    from .in_out_classifier import EnhancedInOutSystem
    from .visualization import (
        FrameAnnotator,
        MiniMap,
        HeatmapGenerator,
        CompositeFrameBuilder,
    )
except ImportError:
    from court_detection import (
        ClassicalCourtDetector,
        DeepCourtDetector,
        SegmentationCourtDetector,
        SIFTCourtMatcher,
        transform_point,
        TENNIS_COURT_CORNERS,
        PICKLEBALL_COURT_CORNERS,
    )
    from object_detection import (
        Detection,
        YOLODetector,
        TrackNetDetector,
        ClassicalPlayerDetector,
        ClassicalBallDetector,
        FasterRCNNDetector,
        non_max_suppression,
    )
    from object_tracking import (
        BallTracker,
        DeepSORTTracker,
        Track,
    )
    from trajectory_3d import (
        TrajectoryReconstructor,
        Trajectory3D,
    )
    from in_out_classifier import EnhancedInOutSystem
    from visualization import (
        FrameAnnotator,
        MiniMap,
        HeatmapGenerator,
        CompositeFrameBuilder,
    )


class PipelineConfig:
    """Configuration container for the pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}

        # Detection method: "yolo", "tracknet", "classical", "combined"
        self.detection_method = self.config.get("detection_method", "combined")

        # Court detection method: "classical" or "deep"
        self.court_method = self.config.get("court_method", "classical")

        # Player detection: "yolo", "fasterrcnn", "hog", "haar"
        self.player_method = self.config.get("player_method", "yolo")

        # YOLO model path
        self.yolo_model = self.config.get("yolo_model", "yolov8s.pt")
        self.yolo_conf = self.config.get("yolo_conf", 0.25)

        # TrackNet weights
        self.tracknet_weights = self.config.get("tracknet_weights", None)

        # Pickleball segmentation model path
        self.seg_model_path = self.config.get(
            "seg_model_path", "models/pickleball_court/best.pt"
        )

        # Court type: "tennis" or "pickleball"
        self.court_type = self.config.get("court_type", "tennis")

        # Max players on court (2 for singles, 4 for doubles)
        self.max_players = self.config.get("max_players", 4)

        # Tracking parameters
        self.process_noise = self.config.get("process_noise", 5.0)
        self.measurement_noise = self.config.get("measurement_noise", 2.0)
        self.max_missing_frames = self.config.get("max_missing_frames", 10)

        # Visualization
        self.show_trajectory = self.config.get("show_trajectory", True)
        self.show_minimap = self.config.get("show_minimap", True)
        self.show_heatmap = self.config.get("show_heatmap", True)
        self.trajectory_length = self.config.get("trajectory_length", 30)

        # Output
        self.output_fps = self.config.get("output_fps", None)  # None = same as input
        self.output_codec = self.config.get("output_codec", "mp4v")

        # Device
        self.device = self.config.get("device", "cuda")


class TennisPickleballPipeline:
    """
    Complete end-to-end video processing pipeline.

    Orchestrates all modules to process a tennis/pickleball video
    and produce annotated output with tracking, trajectory, and
    in/out analysis.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._initialize_modules()

    def _initialize_modules(self):
        """Initialize all pipeline modules based on configuration."""
        cfg = self.config

        # Module 1: Court Detection
        try:
            from .court_detection import load_court_config
        except ImportError:
            from court_detection import load_court_config
        court_cfg = load_court_config("configs/court_config.yaml")
        self.sift_matcher = SIFTCourtMatcher()

        # Select court corner model based on court type
        if cfg.court_type == "pickleball":
            self.court_corners = PICKLEBALL_COURT_CORNERS
            # Use segmentation model for pickleball if available
            if cfg.seg_model_path and os.path.exists(cfg.seg_model_path):
                self.court_detector = SegmentationCourtDetector(
                    model_path=cfg.seg_model_path,
                    conf_threshold=0.3,
                )
                print(f"[Pipeline] Using SegmentationCourtDetector: {cfg.seg_model_path}")
            else:
                self.court_detector = ClassicalCourtDetector(court_cfg)
                print(f"[Pipeline] Seg model not found at '{cfg.seg_model_path}', "
                      f"falling back to ClassicalCourtDetector")
        else:
            self.court_corners = TENNIS_COURT_CORNERS
            self.court_detector = ClassicalCourtDetector(court_cfg)

        # Module 2: Object Detection
        self.ball_detectors = {}
        self.player_detectors = {}

        # Determine if YOLO model is custom-trained or COCO pretrained
        _is_custom = cfg.config.get("is_custom_yolo", False)

        # Ball detection
        if cfg.detection_method in ("yolo", "combined"):
            try:
                self._yolo_detector = YOLODetector(
                    model_path=cfg.yolo_model,
                    conf_threshold=cfg.yolo_conf,
                    device=cfg.device,
                    is_custom_model=_is_custom,
                )
                self.ball_detectors["yolo"] = self._yolo_detector
            except Exception as e:
                self._yolo_detector = None
                print(f"WARNING: Could not load YOLO: {e}")

        if cfg.detection_method in ("tracknet", "combined"):
            self.ball_detectors["tracknet"] = TrackNetDetector(
                weights_path=cfg.tracknet_weights,
                device=cfg.device,
            )

        if cfg.detection_method in ("classical", "combined"):
            self.ball_detectors["classical"] = ClassicalBallDetector()

        # Player detection
        # When using custom ball model, ball YOLO has no 'person' class
        # so we need a separate COCO YOLO for players
        if cfg.player_method == "yolo":
            if _is_custom and "yolo" in self.ball_detectors:
                # Ball model is custom → create separate COCO YOLO for players
                try:
                    self.player_detectors["yolo"] = YOLODetector(
                        model_path="yolov8s.pt",
                        conf_threshold=cfg.yolo_conf,
                        device=cfg.device,
                        is_custom_model=False,  # COCO pretrained
                    )
                    print("[Pipeline] Using separate COCO YOLO for player detection")
                except Exception as e:
                    print(f"WARNING: Could not load player YOLO: {e}")
            elif "yolo" in self.ball_detectors:
                self.player_detectors["yolo"] = self.ball_detectors["yolo"]
        elif cfg.player_method == "fasterrcnn":
            try:
                self.player_detectors["fasterrcnn"] = FasterRCNNDetector(
                    device=cfg.device
                )
            except Exception as e:
                print(f"WARNING: Could not load Faster R-CNN: {e}")
        elif cfg.player_method in ("hog", "haar"):
            self.player_detectors["classical"] = ClassicalPlayerDetector(
                method=cfg.player_method
            )

        # Module 3: Object Tracking
        self.ball_tracker = BallTracker(
            process_noise_std=cfg.process_noise,
            measurement_noise_std=cfg.measurement_noise,
            max_missing_frames=cfg.max_missing_frames,
        )
        self.player_tracker = DeepSORTTracker()

        # Module 4: 3D Reconstruction (initialized when homography is computed)
        self.trajectory_reconstructor: Optional[TrajectoryReconstructor] = None
        self.in_out_system = EnhancedInOutSystem(
            court_type=f"{cfg.court_type}_singles"
        )

        # Visualization
        self.annotator = FrameAnnotator()
        self.minimap = MiniMap(court_type=cfg.court_type)
        self.heatmap_gen = HeatmapGenerator()
        self.composite_builder = CompositeFrameBuilder()

        # State
        self.homography: Optional[np.ndarray] = None
        self.homography_inv: Optional[np.ndarray] = None
        self.prev_valid_homography: Optional[np.ndarray] = None
        self.court_image_polygon: Optional[np.ndarray] = None
        self.frame_size: Optional[Tuple[int, int]] = None  # (width, height)
        self.ball_pixel_positions: List[Tuple[float, float]] = []
        self.frame_count = 0

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        show_preview: bool = False,
        max_frames: Optional[int] = None,
    ) -> Dict:
        """
        Process a complete video file.

        Args:
            input_path: Path to input video
            output_path: Path to output annotated video (None = no output)
            show_preview: Show real-time preview window
            max_frames: Limit processing to N frames (None = all)

        Returns:
            Dictionary with processing results and statistics
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        # Video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        # Output video writer
        writer = None
        if output_path:
            out_fps = self.config.output_fps or fps
            fourcc = cv2.VideoWriter_fourcc(*self.config.output_codec)
            out_w = width + 220  # Extra width for minimap panel
            out_h = height
            writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))

        # Processing loop
        stats = {
            "total_frames": total_frames,
            "fps": fps,
            "resolution": (width, height),
            "ball_detections": 0,
            "player_detections": 0,
            "bounces": [],
            "processing_time": 0,
        }

        start_time = time.time()

        progress = tqdm(total=total_frames, desc="Processing video")

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Process single frame
            result = self.process_frame(frame, frame_idx)

            # Build output frame
            output_frame = self._build_output_frame(frame, result)

            # Write output
            if writer:
                writer.write(output_frame)

            # Preview
            if show_preview:
                preview = cv2.resize(output_frame, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("Tennis/Pickleball Tracker", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Update stats
            if result.get("ball_detected"):
                stats["ball_detections"] += 1
            stats["player_detections"] += len(result.get("player_tracks", []))

            progress.update(1)

        progress.close()

        # Finalize
        elapsed = time.time() - start_time
        stats["processing_time"] = elapsed
        stats["processing_fps"] = total_frames / elapsed if elapsed > 0 else 0

        # Finalize trajectory and get bounce/in-out results
        if self.trajectory_reconstructor:
            trajectory = self.trajectory_reconstructor.finalize()
            analysis = self.in_out_system.analyze_trajectory(trajectory)
            stats["bounces"] = [
                {
                    "frame": b.frame_id,
                    "x": b.x,
                    "y": b.y,
                    "is_in": b.is_in,
                    "confidence": b.confidence,
                }
                for b in analysis["bounces"]
            ]
            stats["analysis_summary"] = analysis["summary"]

        # Generate heatmaps
        if self.config.show_heatmap:
            os.makedirs("outputs/reports", exist_ok=True)
            self.heatmap_gen.get_ball_heatmap_image(
                save_path="outputs/reports/ball_heatmap.png"
            )

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        return stats

    def process_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> Dict:
        """
        Process a single video frame through the full pipeline.

        Returns dictionary with all intermediate and final results.
        """
        self.frame_count = frame_idx
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        result = {
            "frame_idx": frame_idx,
            "ball_detected": False,
            "ball_position": None,
            "ball_tracked_position": None,
            "player_detections": [],
            "player_tracks": [],
            "court_detected": False,
            "homography": None,
        }

        # ---- Module 1: Court Detection ----
        if self.homography is None or frame_idx % 30 == 0:
            H, court_result = self.court_detector.detect_and_compute_homography(
                frame, court_keypoints=self.court_corners
            )

            if H is not None:
                # Check temporal consistency: if we already have a valid
                # homography, reject the new one if it deviates too much
                # (likely a mis-detection on a noisy frame).
                if self.prev_valid_homography is not None:
                    diff = np.linalg.norm(
                        H / H[2, 2] - self.prev_valid_homography / self.prev_valid_homography[2, 2]
                    )
                    if diff > 5.0:
                        H = self.prev_valid_homography
                self.homography = H
                self.prev_valid_homography = H.copy()
            elif self.prev_valid_homography is not None:
                self.homography = self.prev_valid_homography

            if self.homography is not None:
                result["court_detected"] = True
                result["homography"] = self.homography

                court_l = 13.41 if self.config.court_type == "pickleball" else 23.77
                court_w = 6.10 if self.config.court_type == "pickleball" else 10.97
                poly = self._build_court_polygon_from_homography(
                    self.homography, frame.shape,
                    court_length=court_l, court_width=court_w,
                )
                if poly is None and court_result.get("intersections"):
                    poly = self._build_court_polygon(
                        court_result["intersections"], frame.shape
                    )
                if poly is not None:
                    self.court_image_polygon = poly

                if "classical" in self.ball_detectors and self.court_image_polygon is not None:
                    self.ball_detectors["classical"].set_court_polygon(
                        self.court_image_polygon.astype(np.int32)
                    )

                if self.trajectory_reconstructor is None:
                    fps = 30.0
                    self.trajectory_reconstructor = TrajectoryReconstructor(
                        homography=self.homography,
                        fps=fps,
                        court_type=f"{self.config.court_type}_singles",
                    )

        # Report court status even on non-redetection frames
        if self.homography is not None and not result["court_detected"]:
            result["court_detected"] = True
            result["homography"] = self.homography

        # ---- Module 2: Object Detection ----
        ball_detection = None
        player_detections = []

        # Ball detection: try classical first (better for small tennis ball),
        # then YOLO, then other methods
        detection_order = ["classical", "yolo", "tracknet"]
        for name in detection_order:
            detector = self.ball_detectors.get(name)
            if detector is None:
                continue
            dets = detector.detect(frame)
            ball_dets = [d for d in dets if d.class_name in ("ball", "sports ball")]
            if ball_dets:
                # Take highest confidence ball detection
                ball_detection = max(ball_dets, key=lambda d: d.confidence)
                break

        # Player detection
        for name, detector in self.player_detectors.items():
            dets = detector.detect(frame)
            player_detections = [d for d in dets if d.class_name in ("player", "person")]
            if player_detections:
                player_detections = non_max_suppression(player_detections)
                break

        # ---- Filter players to court area only ----
        player_detections = self._filter_players_on_court(player_detections)

        if ball_detection:
            result["ball_detected"] = True
            result["ball_position"] = ball_detection.center
            self.ball_pixel_positions.append(ball_detection.center)

        result["player_detections"] = player_detections

        # ---- Module 3: Object Tracking ----
        # Ball tracking (Kalman + Optical Flow)
        ball_pos, ball_track = self.ball_tracker.update(frame, ball_detection)
        if ball_pos:
            result["ball_tracked_position"] = ball_pos

        # Player tracking (DeepSORT)
        tracked_players = self.player_tracker.update(player_detections, frame)
        result["player_tracks"] = tracked_players

        # ---- Module 4: 3D Reconstruction ----
        if self.trajectory_reconstructor and ball_pos:
            self.trajectory_reconstructor.process_frame(
                frame_idx, ball_pos[0], ball_pos[1]
            )

        # ---- Heatmap accumulation ----
        if self.homography is not None and ball_pos:
            court_pos = transform_point(
                self.homography, np.array([ball_pos[0], ball_pos[1]])
            )
            self.heatmap_gen.add_ball_position(court_pos[0], court_pos[1])
            result["ball_court_position"] = (court_pos[0], court_pos[1])

        return result

    @staticmethod
    def _build_court_polygon_from_homography(
        H: np.ndarray,
        frame_shape: Tuple,
        padding_m: float = 2.0,
        court_length: float = 23.77,
        court_width: float = 10.97,
    ) -> Optional[np.ndarray]:
        """
        Build court polygon by projecting standard court corners through
        the inverse homography (court coords -> image coords).

        This is much more accurate than using raw intersections, since it
        only outlines the *main* court and ignores adjacent courts.
        """
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None

        h, w = frame_shape[:2]

        court_pts = np.array([
            [-padding_m,                  -padding_m],
            [-padding_m,                  court_width + padding_m],
            [court_length + padding_m,    court_width + padding_m],
            [court_length + padding_m,    -padding_m],
        ], dtype=np.float32)

        # Project court corners to image space using cv2.perspectiveTransform
        court_pts_2d = court_pts.reshape(-1, 1, 2)
        img_pts = cv2.perspectiveTransform(court_pts_2d, H_inv)
        img_pts = img_pts.reshape(-1, 2).astype(np.float32)

        # Check for NaN / inf
        if np.any(~np.isfinite(img_pts)):
            return None

        # Sanity check: projected points should be somewhat near the frame
        margin = max(w, h)
        if (np.any(img_pts < -margin) or np.any(img_pts[:, 0] > w + margin)
                or np.any(img_pts[:, 1] > h + margin)):
            return None

        # Extend polygon to frame bottom for near-side players
        x_min = float(img_pts[:, 0].min())
        x_max = float(img_pts[:, 0].max())
        bottom_pts = np.array([
            [max(0.0, x_min), float(h)],
            [min(float(w), x_max), float(h)],
        ], dtype=np.float32)

        all_pts = np.vstack([img_pts, bottom_pts]).astype(np.float32)
        hull = cv2.convexHull(all_pts)
        return hull.reshape(-1, 2).astype(np.float32)

    @staticmethod
    def _build_court_polygon(
        intersections: List[Tuple[float, float]],
        frame_shape: Tuple,
    ) -> Optional[np.ndarray]:
        """
        Build a court polygon from detected line intersections.

        Steps:
        1. Filter intersections to only those within/near the frame
        2. Build convex hull
        3. Extend polygon to frame bottom (near-side baseline is
           often at the very bottom of the camera view)
        """
        h, w = frame_shape[:2]
        margin = int(w * 0.1)  # 10% margin outside frame

        # Step 1: Keep only intersections within frame + margin
        filtered = []
        for pt in intersections:
            if -margin <= pt[0] <= w + margin and -margin <= pt[1] <= h + margin:
                filtered.append(pt)

        if len(filtered) < 3:
            return None

        pts = np.array(filtered, dtype=np.float32)

        # Step 2: Convex hull of filtered points
        hull = cv2.convexHull(pts)
        hull_pts = hull.reshape(-1, 2)

        # Step 3: Extend the polygon to the bottom of the frame.
        # The camera looks down at the court, so the near baseline
        # extends to the bottom of the frame. Add two corner points
        # at the bottom spanning the hull's x-range.
        x_min = float(hull_pts[:, 0].min())
        x_max = float(hull_pts[:, 0].max())
        bottom_pts = np.array([
            [x_min, float(h)],
            [x_max, float(h)],
        ], dtype=np.float32)

        all_pts = np.vstack([hull_pts, bottom_pts])
        final_hull = cv2.convexHull(all_pts)
        return final_hull.reshape(-1, 2).astype(np.float32)

    def _filter_players_on_court(
        self, detections: List[Detection]
    ) -> List[Detection]:
        """
        Filter player detections to only those on or near the court.

        Uses a two-stage approach:
        1. Compute the court x-center from detected court intersections
           or from the median of player x-positions.
        2. Score each detection by court proximity, perspective-
           consistent size, and confidence.  Hard-reject players
           whose center_x is far from the court center.
        3. Keep top max_players.
        """
        if not detections:
            return detections

        frame_w = self.frame_size[0] if self.frame_size else 1920
        frame_h = self.frame_size[1] if self.frame_size else 1080

        # Use frame center as court center (robust for typical
        # broadcast angles where the main court is roughly centered).
        court_cx = frame_w / 2.0
        # Max allowed x-distance: ~40% of frame width
        max_x_dist = frame_w * 0.40

        scored = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2.0
            foot_y = float(y2)
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            area = bbox_w * bbox_h

            # --- Hard reject: too far from court center in x ---
            x_dist = abs(cx - court_cx)
            if x_dist > max_x_dist:
                continue

            # --- Court proximity score (x-distance from court center) ---
            # Normalized so 0 = at center, 1 = at max_x_dist
            x_norm = x_dist / max_x_dist if max_x_dist > 0 else 0
            court_score = 1.0 - x_norm

            # --- Perspective-consistent size score ---
            # Players closer to the camera (larger foot_y) should be
            # bigger.  Compute an expected height based on foot_y and
            # penalize detections that deviate.
            # Near-side (foot_y ~ frame_h) expect height ~ 300+
            # Far-side  (foot_y ~ 0.25*frame_h) expect height ~ 80-120
            expected_h = 50 + (foot_y / frame_h) * 300
            size_ratio = min(bbox_h, expected_h) / max(bbox_h, expected_h)
            size_score = size_ratio  # 0..1, 1 = perfect match

            # --- Confidence ---
            conf_score = det.confidence

            # Combined score
            total = (
                conf_score * 0.15
                + size_score * 0.25
                + court_score * 0.60
            )
            scored.append((total, det))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep top max_players
        max_p = self.config.max_players
        return [s[1] for s in scored[:max_p]]

    def _build_output_frame(
        self, frame: np.ndarray, result: Dict
    ) -> np.ndarray:
        """Build the annotated output frame with all visualizations."""
        annotated = frame.copy()

        # Draw player tracks
        if result["player_tracks"]:
            annotated = self.annotator.draw_tracked_players(
                annotated, result["player_tracks"]
            )

        # Draw court lines overlay (hybrid: seg mask + classical CV)
        if self.homography is not None and hasattr(self, 'court_detector'):
            try:
                from court_detection import detect_court_lines_hybrid, draw_court_lines_overlay
                # Get court seg mask from detector
                model = self.court_detector._get_model()
                preds = model(frame, conf=0.3, verbose=False)
                court_mask = None
                for r in preds:
                    if r.masks is not None and len(r.masks) > 0:
                        mask_np = r.masks[0].data[0].cpu().numpy()
                        h_f, w_f = frame.shape[:2]
                        court_mask = cv2.resize(mask_np, (w_f, h_f))
                        court_mask = (court_mask > 0.5).astype(np.uint8)
                        break
                if court_mask is not None:
                    court_lines = detect_court_lines_hybrid(frame, court_mask)
                    annotated = draw_court_lines_overlay(annotated, court_lines)
                    # Use hybrid corners for more precise homography if available
                    if court_lines["corners"] is not None:
                        from court_detection import PICKLEBALL_COURT_CORNERS
                        H_precise, _ = cv2.findHomography(
                            court_lines["corners"][:4].astype(np.float32),
                            PICKLEBALL_COURT_CORNERS[:4].astype(np.float32),
                            cv2.RANSAC, 5.0,
                        )
                        if H_precise is not None:
                            self.homography = H_precise
            except Exception:
                pass  # Fall back to no overlay

        # Draw ball trajectory
        if self.config.show_trajectory and self.ball_pixel_positions:
            annotated = self.annotator.draw_ball_trajectory(
                annotated,
                self.ball_pixel_positions,
                max_trail=self.config.trajectory_length,
            )

        # Draw ball detection
        if result["ball_detected"] and result["ball_position"]:
            cx, cy = result["ball_position"]
            cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 255, 255), -1)

        # Info overlay
        info = {
            "Frame": str(result["frame_idx"]),
            "Ball": "Detected" if result["ball_detected"] else "Lost",
            "Players": str(len(result["player_tracks"])),
        }
        annotated = self.annotator.draw_info_overlay(annotated, info)

        # Mini-map
        minimap_img = None
        if self.config.show_minimap:
            ball_court = result.get("ball_court_position")

            player_court_positions = None
            if self.homography is not None and result["player_tracks"]:
                player_court_positions = []
                for _track_id, det in result["player_tracks"]:
                    foot_x = (det.bbox[0] + det.bbox[2]) / 2.0
                    foot_y = float(det.bbox[3])
                    court_pos = transform_point(
                        self.homography, np.array([foot_x, foot_y])
                    )
                    player_court_positions.append(
                        (float(court_pos[0]), float(court_pos[1]))
                    )

            minimap_img = self.minimap.render(
                ball_court_pos=ball_court,
                player_court_positions=player_court_positions,
            )

        # Composite
        output = self.composite_builder.build(
            annotated,
            minimap=minimap_img,
            info=info,
        )

        return output


# ============================================================================
# Data Processing Utilities (Week 2)
# ============================================================================

class DataPreprocessor:
    """
    Video data preprocessing utilities.

    - Extract frames from video
    - Resize to standard dimensions
    - Normalize pixel values
    - Split into train/val/test sets
    - Convert label formats
    """

    def __init__(
        self,
        target_size_tracknet: Tuple[int, int] = (640, 360),
        target_size_yolo: Tuple[int, int] = (640, 640),
    ):
        self.target_size_tracknet = target_size_tracknet
        self.target_size_yolo = target_size_yolo

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        frame_interval: int = 1,
        max_frames: Optional[int] = None,
    ) -> int:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            frame_interval: Extract every N-th frame
            max_frames: Maximum number of frames to extract

        Returns:
            Number of frames extracted
        """
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                filename = os.path.join(output_dir, f"frame_{count:06d}.jpg")
                cv2.imwrite(filename, frame)
                count += 1

                if max_frames and count >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        print(f"Extracted {count} frames to {output_dir}")
        return count

    def resize_frames(
        self,
        input_dir: str,
        output_dir: str,
        target_size: Tuple[int, int],
        normalize: bool = False,
    ) -> int:
        """
        Resize all frames in a directory.

        Args:
            input_dir: Directory with source frames
            output_dir: Directory for resized frames
            target_size: (width, height) target dimensions
            normalize: If True, normalize pixel values to [0, 1]

        Returns:
            Number of frames processed
        """
        os.makedirs(output_dir, exist_ok=True)
        count = 0

        for filename in sorted(os.listdir(input_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img = cv2.imread(os.path.join(input_dir, filename))
            if img is None:
                continue

            resized = cv2.resize(img, target_size)

            if normalize:
                resized = resized.astype(np.float32) / 255.0

            output_path = os.path.join(output_dir, filename)
            if normalize:
                np.save(output_path.replace(".jpg", ".npy"), resized)
            else:
                cv2.imwrite(output_path, resized)

            count += 1

        print(f"Resized {count} frames to {target_size} in {output_dir}")
        return count

    @staticmethod
    def split_dataset(
        frames_dir: str,
        labels_dir: str,
        output_base: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Split dataset into train/val/test sets.

        Creates subdirectories under output_base:
        - train/images, train/labels
        - val/images, val/labels
        - test/images, test/labels
        """
        import shutil
        import random

        random.seed(seed)

        # Get sorted file list
        frames = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        random.shuffle(frames)

        n = len(frames)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": frames[:n_train],
            "val": frames[n_train : n_train + n_val],
            "test": frames[n_train + n_val :],
        }

        for split_name, split_frames in splits.items():
            img_dir = os.path.join(output_base, split_name, "images")
            lbl_dir = os.path.join(output_base, split_name, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            for fname in split_frames:
                # Copy image
                src_img = os.path.join(frames_dir, fname)
                dst_img = os.path.join(img_dir, fname)
                shutil.copy2(src_img, dst_img)

                # Copy label (if exists)
                label_name = os.path.splitext(fname)[0] + ".txt"
                src_lbl = os.path.join(labels_dir, label_name)
                if os.path.exists(src_lbl):
                    dst_lbl = os.path.join(lbl_dir, label_name)
                    shutil.copy2(src_lbl, dst_lbl)

            print(f"{split_name}: {len(split_frames)} samples")

    @staticmethod
    def convert_to_yolo_format(
        annotations: List[Dict],
        image_width: int,
        image_height: int,
        output_path: str,
    ):
        """
        Convert annotations to YOLO format.

        YOLO format: class_id x_center y_center width height
        All values normalized to [0, 1] relative to image dimensions.

        Args:
            annotations: List of dicts with keys: class_id, x1, y1, x2, y2
            image_width: Image width in pixels
            image_height: Image height in pixels
            output_path: Path to save .txt label file
        """
        with open(output_path, "w") as f:
            for ann in annotations:
                cls_id = ann["class_id"]
                x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]

                # Convert to YOLO format (normalized center + size)
                x_center = ((x1 + x2) / 2) / image_width
                y_center = ((y1 + y2) / 2) / image_height
                w = (x2 - x1) / image_width
                h = (y2 - y1) / image_height

                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for running the pipeline from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tennis/Pickleball Detection & Tracking Pipeline"
    )
    parser.add_argument(
        "input", type=str, help="Path to input video file"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Path to output video file"
    )
    parser.add_argument(
        "-c", "--config", type=str, default=None,
        help="Path to pipeline config YAML"
    )
    parser.add_argument(
        "--court-type", type=str, default="tennis",
        choices=["tennis", "pickleball"],
        help="Court type"
    )
    parser.add_argument(
        "--detection", type=str, default="combined",
        choices=["yolo", "tracknet", "classical", "combined"],
        help="Ball detection method"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show preview window"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Process only first N frames"
    )

    args = parser.parse_args()

    # Build config
    config = PipelineConfig(args.config)
    config.court_type = args.court_type
    config.detection_method = args.detection

    # Run pipeline
    pipeline = TennisPickleballPipeline(config)

    output = args.output
    if output is None:
        base = os.path.splitext(args.input)[0]
        output = f"{base}_tracked.mp4"

    stats = pipeline.process_video(
        args.input,
        output,
        show_preview=args.show,
        max_frames=args.max_frames,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"Total frames: {stats['total_frames']}")
    print(f"Processing time: {stats['processing_time']:.1f}s")
    print(f"Processing FPS: {stats['processing_fps']:.1f}")
    print(f"Ball detections: {stats['ball_detections']}")
    print(f"Output: {output}")

    if stats.get("bounces"):
        print(f"\nBounce events: {len(stats['bounces'])}")
        for i, b in enumerate(stats["bounces"]):
            status = "IN" if b["is_in"] else "OUT"
            print(
                f"  Bounce {i+1}: frame {b['frame']}, "
                f"({b['x']:.2f}, {b['y']:.2f}) - {status} "
                f"(conf: {b['confidence']:.2f})"
            )


if __name__ == "__main__":
    main()
