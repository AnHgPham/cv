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
        SIFTCourtMatcher,
        transform_point,
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
        SIFTCourtMatcher,
        transform_point,
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

        # Court type: "tennis" or "pickleball"
        self.court_type = self.config.get("court_type", "tennis")

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
        self.court_detector = ClassicalCourtDetector()
        self.sift_matcher = SIFTCourtMatcher()

        # Module 2: Object Detection
        self.ball_detectors = {}
        self.player_detectors = {}

        # Ball detection
        if cfg.detection_method in ("yolo", "combined"):
            try:
                self.ball_detectors["yolo"] = YOLODetector(
                    model_path=cfg.yolo_model,
                    conf_threshold=cfg.yolo_conf,
                    device=cfg.device,
                )
            except Exception as e:
                print(f"WARNING: Could not load YOLO: {e}")

        if cfg.detection_method in ("tracknet", "combined"):
            self.ball_detectors["tracknet"] = TrackNetDetector(
                weights_path=cfg.tracknet_weights,
                device=cfg.device,
            )

        if cfg.detection_method in ("classical", "combined"):
            self.ball_detectors["classical"] = ClassicalBallDetector()

        # Player detection
        if cfg.player_method == "yolo" and "yolo" in self.ball_detectors:
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
            # Re-detect court periodically (handles camera movement)
            H, court_result = self.court_detector.detect_and_compute_homography(
                frame
            )
            if H is not None:
                self.homography = H
                result["court_detected"] = True
                result["homography"] = H

                # Initialize/update 3D reconstructor
                if self.trajectory_reconstructor is None:
                    fps = 30.0  # Default; overridden in process_video
                    self.trajectory_reconstructor = TrajectoryReconstructor(
                        homography=H,
                        fps=fps,
                        court_type=f"{self.config.court_type}_singles",
                    )

        # ---- Module 2: Object Detection ----
        ball_detection = None
        player_detections = []

        # Ball detection
        for name, detector in self.ball_detectors.items():
            dets = detector.detect(frame)
            ball_dets = [d for d in dets if d.class_name == "ball"]
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
            minimap_img = self.minimap.render(
                ball_court_pos=ball_court,
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
