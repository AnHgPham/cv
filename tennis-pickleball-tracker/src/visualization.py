"""
Visualization Module
====================

Provides visualization tools for the tennis/pickleball tracker:

1. Video overlay: Draw detections, trajectories, and annotations on video frames
2. Mini-map: 2D bird's-eye view of the court with ball and player positions
3. Heatmap: Ball landing distribution and player movement heatmap
4. Trajectory plots: 2D and 3D trajectory visualizations
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from typing import List, Tuple, Optional, Dict

try:
    from .object_detection import Detection
    from .object_tracking import Track
    from .trajectory_3d import Trajectory3D, BounceEvent, TrajectoryPoint3D
    from .court_detection import (
        TENNIS_COURT_KEYPOINTS,
        TENNIS_COURT_CORNERS,
        transform_points,
    )
except ImportError:
    from object_detection import Detection
    from object_tracking import Track
    from trajectory_3d import Trajectory3D, BounceEvent, TrajectoryPoint3D
    from court_detection import (
        TENNIS_COURT_KEYPOINTS,
        TENNIS_COURT_CORNERS,
        transform_points,
    )


# ============================================================================
# Color Palette
# ============================================================================

COLORS = {
    "ball": (0, 255, 255),          # Yellow (BGR)
    "player": (0, 255, 0),          # Green
    "court_line": (255, 255, 255),  # White
    "trajectory": (0, 165, 255),    # Orange
    "bounce_in": (0, 255, 0),       # Green
    "bounce_out": (0, 0, 255),      # Red
    "text": (255, 255, 255),        # White
    "text_bg": (0, 0, 0),           # Black
}

PLAYER_COLORS = [
    (255, 100, 100),   # Blue-ish
    (100, 100, 255),   # Red-ish
    (100, 255, 100),   # Green-ish
    (255, 255, 100),   # Cyan-ish
]


# ============================================================================
# Frame Overlay Drawing
# ============================================================================

class FrameAnnotator:
    """
    Draws annotations on video frames: bounding boxes, trajectories,
    labels, and status information.
    """

    def __init__(self, line_thickness: int = 2, font_scale: float = 0.6):
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        draw_labels: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes and labels for detections on frame."""
        annotated = frame.copy()

        for det in detections:
            color = COLORS.get(det.class_name, (128, 128, 128))
            x1, y1, x2, y2 = det.bbox

            # Bounding box
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), color, self.line_thickness
            )

            # Center dot
            cx, cy = int(det.center[0]), int(det.center[1])
            cv2.circle(annotated, (cx, cy), 3, color, -1)

            if draw_labels:
                label = f"{det.class_name} {det.confidence:.2f}"
                self._draw_label(annotated, label, (x1, y1 - 5), color)

        return annotated

    def draw_tracked_players(
        self,
        frame: np.ndarray,
        tracked: List[Tuple[int, Detection]],
    ) -> np.ndarray:
        """Draw tracked players with IDs and color-coded boxes."""
        annotated = frame.copy()

        for track_id, det in tracked:
            tid = int(track_id) if isinstance(track_id, (int, float)) else hash(track_id)
            color = PLAYER_COLORS[tid % len(PLAYER_COLORS)]
            x1, y1, x2, y2 = det.bbox

            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), color, self.line_thickness
            )

            label = f"P{track_id}"
            self._draw_label(annotated, label, (x1, y1 - 5), color)

        return annotated

    def draw_ball_trajectory(
        self,
        frame: np.ndarray,
        positions: List[Tuple[float, float]],
        max_trail: int = 30,
        fade: bool = True,
    ) -> np.ndarray:
        """
        Draw ball trajectory trail on frame.

        Shows the last max_trail positions with fading effect.
        """
        annotated = frame.copy()
        n = min(len(positions), max_trail)
        trail = positions[-n:]

        for i in range(1, len(trail)):
            # Fade: older points are more transparent
            if fade:
                alpha = i / len(trail)
                thickness = max(1, int(self.line_thickness * alpha))
            else:
                alpha = 1.0
                thickness = self.line_thickness

            color = tuple(int(c * alpha) for c in COLORS["trajectory"])
            pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
            pt2 = (int(trail[i][0]), int(trail[i][1]))

            cv2.line(annotated, pt1, pt2, color, thickness)

        # Current position: bright dot
        if trail:
            cx, cy = int(trail[-1][0]), int(trail[-1][1])
            cv2.circle(annotated, (cx, cy), 5, COLORS["ball"], -1)
            cv2.circle(annotated, (cx, cy), 7, COLORS["ball"], 1)

        return annotated

    def draw_bounce_markers(
        self,
        frame: np.ndarray,
        bounces: List[BounceEvent],
        H_inv: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw bounce markers (in=green circle, out=red X) on frame."""
        annotated = frame.copy()

        for bounce in bounces:
            if H_inv is not None:
                # Project court coords back to image
                court_pt = np.array([[bounce.x, bounce.y]], dtype=np.float32)
                img_pt = transform_points(H_inv, court_pt)[0]
                x, y = int(img_pt[0]), int(img_pt[1])
            else:
                x, y = int(bounce.x), int(bounce.y)

            if bounce.is_in:
                cv2.circle(annotated, (x, y), 12, COLORS["bounce_in"], 2)
                self._draw_label(annotated, "IN", (x + 15, y), COLORS["bounce_in"])
            else:
                size = 10
                cv2.line(
                    annotated,
                    (x - size, y - size),
                    (x + size, y + size),
                    COLORS["bounce_out"],
                    2,
                )
                cv2.line(
                    annotated,
                    (x - size, y + size),
                    (x + size, y - size),
                    COLORS["bounce_out"],
                    2,
                )
                self._draw_label(annotated, "OUT", (x + 15, y), COLORS["bounce_out"])

        return annotated

    def draw_info_overlay(
        self,
        frame: np.ndarray,
        info: Dict[str, str],
        position: str = "top_left",
    ) -> np.ndarray:
        """Draw information text overlay on frame."""
        annotated = frame.copy()
        y_offset = 30

        for key, value in info.items():
            text = f"{key}: {value}"
            text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]

            if position == "top_left":
                x = 10
            elif position == "top_right":
                x = frame.shape[1] - text_size[0] - 10
            else:
                x = 10

            # Background rectangle
            cv2.rectangle(
                annotated,
                (x - 2, y_offset - text_size[1] - 5),
                (x + text_size[0] + 2, y_offset + 5),
                COLORS["text_bg"],
                -1,
            )
            cv2.putText(
                annotated,
                text,
                (x, y_offset),
                self.font,
                self.font_scale,
                COLORS["text"],
                1,
            )
            y_offset += text_size[1] + 15

        return annotated

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
    ):
        """Draw a text label with background."""
        text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]
        x, y = position
        cv2.rectangle(
            frame,
            (x, y - text_size[1] - 4),
            (x + text_size[0], y + 2),
            COLORS["text_bg"],
            -1,
        )
        cv2.putText(
            frame, text, (x, y), self.font, self.font_scale, color, 1
        )


# ============================================================================
# Mini-Map (2D Court View)
# ============================================================================

class MiniMap:
    """
    Generates a 2D bird's-eye view mini-map of the court
    showing ball and player positions in real time.
    """

    def __init__(
        self,
        court_type: str = "tennis",
        map_width: int = 300,
        map_height: int = None,
        margin: int = 30,
    ):
        self.court_type = court_type
        self.map_width = map_width
        self.margin = margin

        # Court dimensions in meters
        if court_type == "tennis":
            self.court_length = 23.77
            self.court_width = 10.97
        else:  # pickleball
            self.court_length = 13.41
            self.court_width = 6.10

        # Auto-calculate height from real court aspect ratio
        usable_w = map_width - 2 * margin
        court_ratio = self.court_length / self.court_width
        usable_h = int(usable_w * court_ratio)
        self.map_height = usable_h + 2 * margin if map_height is None else map_height

        # Scale: court coords -> map pixel coords
        usable_h_final = self.map_height - 2 * margin
        self.scale_x = usable_h_final / self.court_length
        self.scale_y = usable_w / self.court_width

        # Background court image (cached)
        self._base_map = self._draw_court()

    def _court_to_pixel(
        self, court_x: float, court_y: float
    ) -> Tuple[int, int]:
        """Convert court coordinates (meters) to mini-map pixel coordinates."""
        px = int(self.margin + court_y * self.scale_y)
        py = int(self.margin + court_x * self.scale_x)
        return (px, py)

    def _draw_court(self) -> np.ndarray:
        """Draw the base court layout (lines, service areas, etc.)."""
        img = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        img[:] = (40, 80, 40)  # Dark green background

        color = (255, 255, 255)
        thickness = 1

        if self.court_type == "tennis":
            # Outer rectangle (doubles)
            tl = self._court_to_pixel(0, 0)
            br = self._court_to_pixel(23.77, 10.97)
            cv2.rectangle(img, tl, br, color, thickness)

            # Singles sidelines
            tl_s = self._court_to_pixel(0, 1.37)
            br_s = self._court_to_pixel(23.77, 9.60)
            cv2.rectangle(img, tl_s, br_s, color, thickness)

            # Service lines
            sl1 = self._court_to_pixel(6.40, 1.37)
            sr1 = self._court_to_pixel(6.40, 9.60)
            cv2.line(img, sl1, sr1, color, thickness)

            sl2 = self._court_to_pixel(17.37, 1.37)
            sr2 = self._court_to_pixel(17.37, 9.60)
            cv2.line(img, sl2, sr2, color, thickness)

            # Center service line
            ct = self._court_to_pixel(6.40, 5.485)
            cb = self._court_to_pixel(17.37, 5.485)
            cv2.line(img, ct, cb, color, thickness)

            # Net
            nl = self._court_to_pixel(11.885, 0)
            nr = self._court_to_pixel(11.885, 10.97)
            cv2.line(img, nl, nr, (200, 200, 200), 2)

        else:  # pickleball
            # Outer rectangle
            tl = self._court_to_pixel(0, 0)
            br = self._court_to_pixel(13.41, 6.10)
            cv2.rectangle(img, tl, br, color, thickness)

            # Kitchen lines (non-volley zone)
            kl1 = self._court_to_pixel(2.13, 0)
            kr1 = self._court_to_pixel(2.13, 6.10)
            cv2.line(img, kl1, kr1, color, thickness)

            kl2 = self._court_to_pixel(11.28, 0)
            kr2 = self._court_to_pixel(11.28, 6.10)
            cv2.line(img, kl2, kr2, color, thickness)

            # Center line
            ct = self._court_to_pixel(2.13, 3.05)
            cb = self._court_to_pixel(11.28, 3.05)
            cv2.line(img, ct, cb, color, thickness)

            # Net
            nl = self._court_to_pixel(6.705, 0)
            nr = self._court_to_pixel(6.705, 6.10)
            cv2.line(img, nl, nr, (200, 200, 200), 2)

        return img

    def render(
        self,
        ball_court_pos: Optional[Tuple[float, float]] = None,
        player_court_positions: Optional[List[Tuple[float, float]]] = None,
        ball_trail: Optional[List[Tuple[float, float]]] = None,
        bounces: Optional[List[BounceEvent]] = None,
    ) -> np.ndarray:
        """
        Render mini-map with current positions.

        Args:
            ball_court_pos: (x, y) ball position in court coords
            player_court_positions: List of (x, y) player positions
            ball_trail: Recent ball positions for trail
            bounces: Bounce events to mark

        Returns:
            Mini-map image (H, W, 3) BGR
        """
        img = self._base_map.copy()

        # Draw ball trail
        if ball_trail:
            for i in range(1, len(ball_trail)):
                p1 = self._court_to_pixel(*ball_trail[i - 1])
                p2 = self._court_to_pixel(*ball_trail[i])
                alpha = i / len(ball_trail)
                color = (
                    int(0 * alpha),
                    int(165 * alpha),
                    int(255 * alpha),
                )
                cv2.line(img, p1, p2, color, 1)

        # Draw bounce markers
        if bounces:
            for bounce in bounces:
                pos = self._court_to_pixel(bounce.x, bounce.y)
                if bounce.is_in:
                    cv2.circle(img, pos, 5, COLORS["bounce_in"], -1)
                else:
                    cv2.circle(img, pos, 5, COLORS["bounce_out"], -1)

        # Draw players
        if player_court_positions:
            for i, (px, py) in enumerate(player_court_positions):
                pos = self._court_to_pixel(px, py)
                color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
                cv2.circle(img, pos, 6, color, -1)
                cv2.circle(img, pos, 7, (255, 255, 255), 1)

        # Draw ball
        if ball_court_pos:
            pos = self._court_to_pixel(*ball_court_pos)
            cv2.circle(img, pos, 4, COLORS["ball"], -1)
            cv2.circle(img, pos, 5, (255, 255, 255), 1)

        return img


# ============================================================================
# Heatmap Generator
# ============================================================================

class HeatmapGenerator:
    """
    Generates heatmaps for ball landing positions and player movement areas.
    """

    def __init__(
        self,
        court_length: float = 23.77,
        court_width: float = 10.97,
        resolution: float = 0.1,  # meters per bin
    ):
        self.court_length = court_length
        self.court_width = court_width
        self.resolution = resolution

        self.bins_x = int(court_length / resolution)
        self.bins_y = int(court_width / resolution)

        # Accumulator grids
        self.ball_heatmap = np.zeros((self.bins_x, self.bins_y))
        self.player_heatmaps: Dict[int, np.ndarray] = {}

    def add_ball_position(self, court_x: float, court_y: float):
        """Add a ball position to the heatmap accumulator."""
        bx = int(np.clip(court_x / self.resolution, 0, self.bins_x - 1))
        by = int(np.clip(court_y / self.resolution, 0, self.bins_y - 1))
        self.ball_heatmap[bx, by] += 1

    def add_player_position(
        self, player_id: int, court_x: float, court_y: float
    ):
        """Add a player position to their heatmap accumulator."""
        if player_id not in self.player_heatmaps:
            self.player_heatmaps[player_id] = np.zeros(
                (self.bins_x, self.bins_y)
            )
        bx = int(np.clip(court_x / self.resolution, 0, self.bins_x - 1))
        by = int(np.clip(court_y / self.resolution, 0, self.bins_y - 1))
        self.player_heatmaps[player_id][bx, by] += 1

    def render_heatmap(
        self,
        heatmap: np.ndarray,
        title: str = "Heatmap",
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Render a heatmap as a matplotlib figure.

        Returns the figure as a BGR image array.
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 12))

        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(heatmap.T, sigma=2)

        ax.imshow(
            smoothed,
            cmap="hot",
            interpolation="bilinear",
            origin="lower",
            extent=[0, self.court_length, 0, self.court_width],
            aspect="auto",
        )

        # Draw court lines overlay
        ax.axhline(y=1.37, color="white", linewidth=0.5, alpha=0.5)
        ax.axhline(y=9.60, color="white", linewidth=0.5, alpha=0.5)
        ax.axvline(x=6.40, color="white", linewidth=0.5, alpha=0.5)
        ax.axvline(x=17.37, color="white", linewidth=0.5, alpha=0.5)
        ax.axvline(x=11.885, color="white", linewidth=1.0, alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("Court Length (m)")
        ax.set_ylabel("Court Width (m)")

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")

        # Convert figure to image array (compatible with all matplotlib versions)
        fig.canvas.draw()
        img_rgba = np.asarray(fig.canvas.buffer_rgba())
        img_rgb = img_rgba[:, :, :3]  # drop alpha channel
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        plt.close(fig)
        return img_bgr

    def get_ball_heatmap_image(
        self, save_path: Optional[str] = None
    ) -> np.ndarray:
        """Render ball position heatmap."""
        return self.render_heatmap(
            self.ball_heatmap,
            title="Ball Position Heatmap",
            save_path=save_path,
        )

    def get_player_heatmap_image(
        self, player_id: int, save_path: Optional[str] = None
    ) -> np.ndarray:
        """Render player movement heatmap."""
        heatmap = self.player_heatmaps.get(
            player_id, np.zeros((self.bins_x, self.bins_y))
        )
        return self.render_heatmap(
            heatmap,
            title=f"Player {player_id} Movement Heatmap",
            save_path=save_path,
        )

    def reset(self):
        """Reset all heatmaps."""
        self.ball_heatmap = np.zeros((self.bins_x, self.bins_y))
        self.player_heatmaps.clear()


# ============================================================================
# Composite Frame Builder
# ============================================================================

class CompositeFrameBuilder:
    """
    Builds a composite output frame combining:
    - Main video with annotations
    - Mini-map
    - Info panel
    """

    def __init__(
        self,
        main_width: int = 1280,
        main_height: int = 720,
        minimap_width: int = 200,
        minimap_height: int = 400,
    ):
        self.main_w = main_width
        self.main_h = main_height
        self.minimap_w = minimap_width
        self.minimap_h = minimap_height

    def build(
        self,
        main_frame: np.ndarray,
        minimap: Optional[np.ndarray] = None,
        info: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """
        Build composite frame.

        Layout:
        +------------------+--------+
        |                  | MiniMap|
        |   Main Video     |--------|
        |                  |  Info  |
        +------------------+--------+
        """
        # Resize main frame
        main = cv2.resize(main_frame, (self.main_w, self.main_h))

        # Create right panel
        panel_h = self.main_h
        panel_w = self.minimap_w + 20
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # Dark gray

        # Add mini-map to panel
        if minimap is not None:
            mm = cv2.resize(minimap, (self.minimap_w, self.minimap_h))
            y_start = 10
            panel[y_start : y_start + self.minimap_h, 10 : 10 + self.minimap_w] = mm

        # Add info text
        if info:
            y = self.minimap_h + 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            for key, value in info.items():
                text = f"{key}: {value}"
                cv2.putText(panel, text, (10, y), font, 0.4, (200, 200, 200), 1)
                y += 20

        # Concatenate horizontally
        composite = np.hstack([main, panel])
        return composite
