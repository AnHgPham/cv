"""
Module 1: Court Detection & Homography Estimation
==================================================

Two approaches implemented:
1. Classical: HSV thresholding -> Morphology -> Canny -> Hough Lines -> Homography
2. Deep Learning: CNN keypoint detection -> Homography

Knowledge applied:
- Image Processing (color spaces, thresholding, morphology)
- Hough Transform (line detection)
- SIFT (feature matching for camera motion)
- Homography (perspective transform)
- CNN (keypoint regression)
"""

import cv2
import numpy as np
import yaml
import os
from typing import List, Tuple, Optional, Dict

# Lazy imports for torch/torchvision (may not be available or may crash on some Python versions)
HAS_TORCH = False
HAS_TORCHVISION = False
torch = None
nn = None
models = None

def _lazy_import_torch():
    """Lazily import torch and torchvision to avoid crashes at module load time."""
    global HAS_TORCH, HAS_TORCHVISION, torch, nn, models
    if torch is not None:
        return
    try:
        import torch as _torch
        import torch.nn as _nn
        torch = _torch
        nn = _nn
        HAS_TORCH = True
    except ImportError:
        return
    try:
        import torchvision.models as _models
        models = _models
        HAS_TORCHVISION = True
    except (ImportError, OSError):
        pass


# ============================================================================
# Court Template: Standard keypoint coordinates (meters)
# ============================================================================

TENNIS_COURT_KEYPOINTS = np.array([
    [0.0, 0.0],        # 0: Top-left baseline corner
    [0.0, 4.115],      # 1: Top-left singles sideline
    [0.0, 5.485],      # 2: Top center baseline
    [0.0, 6.855],      # 3: Top-right singles sideline
    [0.0, 10.97],      # 4: Top-right baseline corner
    [6.40, 4.115],     # 5: Left service line (left singles)
    [6.40, 5.485],     # 6: Service T (left)
    [6.40, 6.855],     # 7: Left service line (right singles)
    [11.885, 0.0],     # 8: Net left
    [11.885, 4.115],   # 9: Net left singles
    [11.885, 5.485],   # 10: Net center
    [11.885, 6.855],   # 11: Net right singles
    [11.885, 10.97],   # 12: Net right
    [17.385, 4.115],   # 13: Right service line (left singles)
    [17.385, 5.485],   # 14: Service T (right)
    [17.385, 6.855],   # 15: Right service line (right singles)
    [23.77, 0.0],      # 16: Bottom-left baseline corner
    [23.77, 4.115],    # 17: Bottom-left singles sideline
    [23.77, 5.485],    # 18: Bottom center baseline
    [23.77, 6.855],    # 19: Bottom-right singles sideline
    [23.77, 10.97],    # 20: Bottom-right baseline corner
], dtype=np.float32)

# Simplified 4-corner keypoints for basic homography
TENNIS_COURT_CORNERS = np.array([
    [0.0, 0.0],        # Top-left
    [0.0, 10.97],      # Top-right
    [23.77, 0.0],      # Bottom-left
    [23.77, 10.97],    # Bottom-right
], dtype=np.float32)

PICKLEBALL_COURT_KEYPOINTS = np.array([
    [0.0, 0.0],        # Top-left
    [0.0, 6.10],       # Top-right
    [2.13, 0.0],       # Kitchen line left (top)
    [2.13, 6.10],      # Kitchen line right (top)
    [6.705, 0.0],      # Net left
    [6.705, 3.05],     # Net center
    [6.705, 6.10],     # Net right
    [11.28, 0.0],      # Kitchen line left (bottom)
    [11.28, 6.10],     # Kitchen line right (bottom)
    [13.41, 0.0],      # Bottom-left
    [13.41, 6.10],     # Bottom-right
], dtype=np.float32)

# 12-keypoint real-world coordinates (meters) for precise homography.
# Order matches detect_court_lines_hybrid keypoint generation:
#   0-3: baseline × sideline (4 corners)
#   4-7: kitchen × sideline (far_left, far_right, near_left, near_right)
#   8-9: baseline × center (far, near)
#   10-11: kitchen × center (far, near)
# Pickleball: 13.41m × 6.10m, kitchen at 2.13m from net, net at 6.705m
PICKLEBALL_KEYPOINTS_12 = np.array([
    [0.0,    0.0],    # KP0: far baseline × left sideline
    [0.0,    6.10],   # KP1: far baseline × right sideline
    [13.41,  0.0],    # KP2: near baseline × left sideline
    [13.41,  6.10],   # KP3: near baseline × right sideline
    [4.575,  0.0],    # KP4: far kitchen × left sideline
    [4.575,  6.10],   # KP5: far kitchen × right sideline
    [8.835,  0.0],    # KP6: near kitchen × left sideline
    [8.835,  6.10],   # KP7: near kitchen × right sideline
    [0.0,    3.05],   # KP8: far baseline × center
    [13.41,  3.05],   # KP9: near baseline × center
    [4.575,  3.05],   # KP10: far kitchen × center
    [8.835,  3.05],   # KP11: near kitchen × center
], dtype=np.float32)

# Backward-compatible 4-corner subset (used by SegmentationCourtDetector)
PICKLEBALL_COURT_CORNERS = PICKLEBALL_KEYPOINTS_12[:4].copy()


def compute_homography_from_keypoints(image_kps, world_kps=None):
    """Compute homography from detected keypoints to world coordinates.
    
    Uses all available matching pairs (≥4 needed) for robust estimation.
    
    Args:
        image_kps: list of (x, y) tuples — detected keypoints in image
        world_kps: array of world coordinates (default: PICKLEBALL_KEYPOINTS_12)
    
    Returns:
        Homography matrix or None
    """
    if world_kps is None:
        world_kps = PICKLEBALL_KEYPOINTS_12
    
    n = min(len(image_kps), len(world_kps))
    if n < 4:
        return None
    
    src = np.array(image_kps[:n], dtype=np.float32)
    dst = world_kps[:n].astype(np.float32)
    
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H


def load_court_config(config_path: str = "configs/court_config.yaml") -> dict:
    """Load court detection configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


# ============================================================================
# Classical Court Detection Pipeline
# ============================================================================

class ClassicalCourtDetector:
    """
    Court detection using classical computer vision techniques.

    Pipeline:
    1. BGR -> HSV color space conversion
    2. White color thresholding (isolate court lines)
    3. Morphological operations (clean noise)
    4. Canny edge detection
    5. Hough Line Transform (detect line segments)
    6. Line classification (horizontal / vertical)
    7. Intersection computation
    8. Homography estimation via RANSAC
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        classical = cfg.get("classical", {})

        # White line thresholds in HSV
        self.white_lower = np.array(
            classical.get("white_lower_hsv", [0, 0, 180])
        )
        self.white_upper = np.array(
            classical.get("white_upper_hsv", [180, 50, 255])
        )

        # Morphology
        self.morph_kernel_size = classical.get("morph_kernel_size", 5)
        self.dilate_iter = classical.get("dilate_iterations", 2)
        self.erode_iter = classical.get("erode_iterations", 1)

        # Canny
        self.canny_low = classical.get("canny_low", 50)
        self.canny_high = classical.get("canny_high", 150)

        # Hough Transform
        self.hough_rho = classical.get("hough_rho", 1)
        self.hough_theta = classical.get("hough_theta", np.pi / 180)
        self.hough_threshold = classical.get("hough_threshold", 100)
        self.hough_min_line_length = classical.get("hough_min_line_length", 50)
        self.hough_max_line_gap = classical.get("hough_max_line_gap", 30)

        # Line classification
        self.h_angle_thresh = classical.get("horizontal_angle_thresh", 30)
        self.v_angle_thresh = classical.get("vertical_angle_thresh", 30)

        # RANSAC
        self.ransac_thresh = classical.get("ransac_reproj_threshold", 5.0)

        # Court surface color (for fallback surface-based detection)
        self.surface_lower = np.array(
            classical.get("surface_lower_hsv", [85, 40, 80])
        )
        self.surface_upper = np.array(
            classical.get("surface_upper_hsv", [130, 255, 255])
        )

        # Manual corners: pixel coordinates [TL, TR, BL, BR] as a
        # fallback when automatic detection cannot isolate the main court.
        manual = classical.get("manual_corners", None)
        self.manual_corners = (
            np.array(manual, dtype=np.float32) if manual else None
        )

        # ROI: fraction of frame height to ignore from top (removes
        # scoreboards, banners in indoor scenes).  0.0 = no crop.
        self.roi_top_crop = classical.get("roi_top_crop", 0.0)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Step 1-3: Color thresholding + morphological cleaning.

        Converts to HSV, thresholds for white pixels (court lines),
        then applies dilation/erosion to clean up noise.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.white_lower, self.white_upper)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        mask = cv2.dilate(mask, kernel, iterations=self.dilate_iter)
        mask = cv2.erode(mask, kernel, iterations=self.erode_iter)

        return mask

    def detect_edges(self, mask: np.ndarray) -> np.ndarray:
        """Step 4: Canny edge detection on the binary mask."""
        edges = cv2.Canny(mask, self.canny_low, self.canny_high)
        return edges

    def detect_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """
        Step 5: Probabilistic Hough Transform to find line segments.

        Returns array of line segments [[x1, y1, x2, y2], ...] or None.
        """
        lines = cv2.HoughLinesP(
            edges,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )
        return lines

    def classify_lines(
        self, lines: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Step 6: Classify lines into horizontal and vertical groups
        based on their angle.

        Returns:
            horizontal_lines: Lines within h_angle_thresh of horizontal
            vertical_lines: Lines within v_angle_thresh of vertical
        """
        horizontal = []
        vertical = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

            if angle < self.h_angle_thresh:
                horizontal.append(line[0])
            elif angle > (90 - self.v_angle_thresh):
                vertical.append(line[0])

        return horizontal, vertical

    @staticmethod
    def merge_similar_lines(
        lines: List[np.ndarray],
        distance_thresh: float = 30.0,
        use_x: bool = False,
    ) -> List[np.ndarray]:
        """
        Merge lines that are close together (likely the same court line).

        Groups lines by proximity and averages each group into one line.

        Args:
            use_x: If True, merge by x-midpoint (for vertical lines).
                   If False, merge by y-midpoint (for horizontal lines).
        """
        if not lines:
            return []

        if use_x:
            mid_fn = lambda l: (l[0] + l[2]) / 2
        else:
            mid_fn = lambda l: (l[1] + l[3]) / 2

        lines = sorted(lines, key=mid_fn)
        merged = []
        current_group = [lines[0]]

        for i in range(1, len(lines)):
            mid_prev = mid_fn(current_group[-1])
            mid_curr = mid_fn(lines[i])

            if abs(mid_curr - mid_prev) < distance_thresh:
                current_group.append(lines[i])
            else:
                avg_line = np.mean(current_group, axis=0).astype(int)
                merged.append(avg_line)
                current_group = [lines[i]]

        avg_line = np.mean(current_group, axis=0).astype(int)
        merged.append(avg_line)

        return merged

    @staticmethod
    def find_intersections(
        horizontal: List[np.ndarray], vertical: List[np.ndarray]
    ) -> List[Tuple[float, float]]:
        """
        Step 7: Find intersection points between horizontal and vertical lines.

        Uses the cross product of homogeneous line representations.
        """
        intersections = []

        for h_line in horizontal:
            x1, y1, x2, y2 = h_line
            h_p1 = np.array([x1, y1, 1.0])
            h_p2 = np.array([x2, y2, 1.0])
            h_l = np.cross(h_p1, h_p2)  # homogeneous line

            for v_line in vertical:
                x3, y3, x4, y4 = v_line
                v_p1 = np.array([x3, y3, 1.0])
                v_p2 = np.array([x4, y4, 1.0])
                v_l = np.cross(v_p1, v_p2)

                pt = np.cross(h_l, v_l)
                if abs(pt[2]) > 1e-8:
                    pt = pt / pt[2]
                    intersections.append((pt[0], pt[1]))

        return intersections

    def compute_homography(
        self,
        image_points: np.ndarray,
        court_points: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Step 8: Compute homography matrix H using RANSAC.

        Maps image pixel coordinates to real-world court coordinates.
        Requires at least 4 point correspondences.

        Args:
            image_points: Nx2 array of pixel coordinates
            court_points: Nx2 array of court coordinates (meters)

        Returns:
            3x3 homography matrix H, or None if computation fails
        """
        if len(image_points) < 4 or len(court_points) < 4:
            return None

        H, mask = cv2.findHomography(
            image_points.astype(np.float32),
            court_points.astype(np.float32),
            cv2.RANSAC,
            self.ransac_thresh,
        )
        return H

    def detect(
        self, frame: np.ndarray
    ) -> Dict:
        """
        Run the full classical court detection pipeline on a single frame.

        Returns:
            Dictionary with keys:
            - 'mask': binary mask of court lines
            - 'edges': Canny edge image
            - 'lines': detected Hough lines
            - 'horizontal': horizontal lines
            - 'vertical': vertical lines
            - 'intersections': list of intersection points
            - 'homography': 3x3 matrix H (may be None)
        """
        result = {
            "mask": None,
            "edges": None,
            "lines": None,
            "horizontal": [],
            "vertical": [],
            "intersections": [],
            "homography": None,
        }

        # Step 1-3: Preprocess
        h_frame, w_frame = frame.shape[:2]
        work = frame
        y_offset = 0
        if self.roi_top_crop > 0:
            y_offset = int(h_frame * self.roi_top_crop)
            work = frame[y_offset:, :]

        mask = self.preprocess(work)
        result["mask"] = mask

        # Step 4: Edge detection
        edges = self.detect_edges(mask)
        result["edges"] = edges

        # Step 5: Hough lines
        raw_lines = self.detect_lines(edges)
        if raw_lines is None or len(raw_lines) == 0:
            return result

        # Shift line coordinates back to full-frame if ROI was applied
        if y_offset > 0:
            raw_lines = raw_lines.copy()
            raw_lines[:, 0, 1] += y_offset
            raw_lines[:, 0, 3] += y_offset
        result["lines"] = raw_lines

        # Step 6: Classify lines
        horizontal, vertical = self.classify_lines(raw_lines)

        # Merge similar lines (horizontal by y-midpoint, vertical by x-midpoint)
        horizontal = self.merge_similar_lines(horizontal, distance_thresh=30, use_x=False)
        vertical = self.merge_similar_lines(vertical, distance_thresh=30, use_x=True)
        result["horizontal"] = horizontal
        result["vertical"] = vertical

        # Step 7: Find intersections
        intersections = self.find_intersections(horizontal, vertical)
        result["intersections"] = intersections

        return result

    def detect_and_compute_homography(
        self,
        frame: np.ndarray,
        court_keypoints: np.ndarray = TENNIS_COURT_CORNERS,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Detect court lines and compute homography to court coordinates.

        Uses a two-pass strategy: first try with standard Hough parameters,
        then retry with relaxed thresholds if insufficient lines are found.
        Validates the resulting homography for geometric plausibility.

        Returns:
            (homography_matrix, detection_result_dict)
        """
        result = self.detect(frame)
        H, result = self._try_homography(result, frame, court_keypoints)

        if H is not None:
            return H, result

        # Retry with relaxed Hough parameters to capture fainter baselines
        relaxed_thresh = max(40, self.hough_threshold - 40)
        relaxed_min_len = max(25, self.hough_min_line_length - 25)
        saved = (self.hough_threshold, self.hough_min_line_length)
        self.hough_threshold = relaxed_thresh
        self.hough_min_line_length = relaxed_min_len

        result2 = self.detect(frame)
        H2, result2 = self._try_homography(result2, frame, court_keypoints)

        self.hough_threshold, self.hough_min_line_length = saved

        if H2 is not None:
            return H2, result2

        # Third pass: detect court by surface color (robust in multi-court scenes)
        surface_corners = self.detect_court_surface(frame)
        if surface_corners is not None and len(surface_corners) >= 4:
            H3 = self.compute_homography(
                surface_corners[:4], court_keypoints[:4]
            )
            if self._validate_homography(H3, frame.shape, court_keypoints[:4]):
                result["homography"] = H3
                result["selected_corners"] = surface_corners
                return H3, result

        # Final fallback: manual corners from config (for known camera setups)
        if self.manual_corners is not None and len(self.manual_corners) >= 4:
            H4 = self.compute_homography(
                self.manual_corners[:4], court_keypoints[:4]
            )
            if H4 is not None:
                result["homography"] = H4
                result["selected_corners"] = self.manual_corners
                return H4, result

        return None, result

    def _try_homography(
        self,
        result: Dict,
        frame: np.ndarray,
        court_keypoints: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """Attempt corner selection + homography from a detection result."""
        intersections = result["intersections"]
        if len(intersections) < 4:
            return None, result

        pts = np.array(intersections, dtype=np.float32)
        corners = self._select_court_corners(pts, frame.shape)

        if corners is not None and len(corners) >= 4:
            H = self.compute_homography(corners[:4], court_keypoints[:4])
            if self._validate_homography(H, frame.shape, court_keypoints[:4]):
                result["homography"] = H
                result["selected_corners"] = corners
                return H, result

        return None, result

    @staticmethod
    def _validate_homography(
        H: Optional[np.ndarray],
        frame_shape: Tuple,
        court_corners: np.ndarray = TENNIS_COURT_CORNERS,
    ) -> bool:
        """
        Validate a homography matrix for geometric plausibility.

        Checks:
        1. H is finite and invertible
        2. Projected court polygon lies near the frame
        3. Projected polygon is convex (4-vertex hull)
        4. Projected area is between 5% and 90% of frame
        5. Round-trip reprojection error < 0.5 m
        """
        if H is None:
            return False
        if not np.all(np.isfinite(H)):
            return False
        if abs(np.linalg.det(H)) < 1e-8:
            return False

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return False

        h, w = frame_shape[:2]
        court_pts = court_corners.reshape(-1, 1, 2).astype(np.float32)

        img_pts = cv2.perspectiveTransform(court_pts, H_inv).reshape(-1, 2)
        if not np.all(np.isfinite(img_pts)):
            return False

        max_range = max(w, h) * 2
        if (np.any(img_pts < -max_range)
                or np.any(img_pts[:, 0] > w + max_range)
                or np.any(img_pts[:, 1] > h + max_range)):
            return False

        hull = cv2.convexHull(img_pts.astype(np.float32))
        if len(hull) != 4:
            return False

        area = cv2.contourArea(hull)
        frame_area = h * w
        if area < frame_area * 0.05 or area > frame_area * 0.90:
            return False

        # Span check: projected court should cover a reasonable fraction
        # of the frame in both x and y (rejects degenerate narrow strips).
        width_span = float(img_pts[:, 0].max() - img_pts[:, 0].min())
        height_span = float(img_pts[:, 1].max() - img_pts[:, 1].min())
        if width_span < w * 0.25 or height_span < h * 0.25:
            return False

        # Perspective ratio: the far side (TL-TR, smaller y) should be at
        # least 15% as wide as the near side (BL-BR, larger y).  Rejects
        # degenerate triangles from clustered center-line intersections.
        # img_pts order: [TL, TR, BL, BR] matching TENNIS_COURT_CORNERS.
        if len(img_pts) == 4:
            far_width = float(np.linalg.norm(img_pts[1] - img_pts[0]))
            near_width = float(np.linalg.norm(img_pts[3] - img_pts[2]))
            if near_width > 1.0:
                ratio = far_width / near_width
                if ratio < 0.15 or ratio > 3.0:
                    return False

        reprojected = cv2.perspectiveTransform(
            cv2.perspectiveTransform(court_pts, H_inv), H
        ).reshape(-1, 2)
        reproj_error = float(np.mean(
            np.linalg.norm(reprojected - court_corners, axis=1)
        ))
        if reproj_error > 0.5:
            return False

        return True

    @staticmethod
    def _select_court_corners(
        points: np.ndarray, frame_shape: Tuple
    ) -> Optional[np.ndarray]:
        """
        Select 4 court corner points from candidate intersections.

        Strategy:
        1. Filter points to frame boundaries.
        2. Compute convex hull; reject if hull area is too small
           (intersections clustered in a narrow strip).
        3. Among hull vertices, pick the 4 that maximize quadrilateral area.
        4. Order as TL, TR, BL, BR for homography correspondence.
        """
        if len(points) < 4:
            return None

        h, w = frame_shape[:2]
        margin = 50

        # Keep only points near the frame
        mask = (
            (points[:, 0] >= -margin) & (points[:, 0] <= w + margin)
            & (points[:, 1] >= -margin) & (points[:, 1] <= h + margin)
        )
        pts = points[mask]
        if len(pts) < 4:
            return None

        hull = cv2.convexHull(pts.astype(np.float32))
        hull_pts = hull.reshape(-1, 2)
        if len(hull_pts) < 4:
            return None

        hull_area = cv2.contourArea(hull)
        if hull_area < h * w * 0.01:
            return None

        # Find 4 hull vertices forming the largest-area quadrilateral
        from itertools import combinations

        best_area = 0.0
        best_quad = None
        n = len(hull_pts)

        for indices in combinations(range(n), 4):
            quad = hull_pts[list(indices)]
            cx_q, cy_q = quad.mean(axis=0)
            angles = np.arctan2(quad[:, 1] - cy_q, quad[:, 0] - cx_q)
            order = np.argsort(angles)
            quad = quad[order]
            area = 0.5 * abs(float(
                np.sum(
                    quad[:, 0] * np.roll(quad[:, 1], -1)
                    - np.roll(quad[:, 0], -1) * quad[:, 1]
                )
            ))
            if area > best_area:
                best_area = area
                best_quad = quad.copy()

        if best_quad is None or best_area < h * w * 0.01:
            return None

        # Order as TL, TR, BL, BR
        sums = best_quad[:, 0] + best_quad[:, 1]
        diffs = best_quad[:, 1] - best_quad[:, 0]
        tl_idx = int(np.argmin(sums))
        br_idx = int(np.argmax(sums))
        tr_idx = int(np.argmin(diffs))
        bl_idx = int(np.argmax(diffs))

        if len({tl_idx, tr_idx, bl_idx, br_idx}) == 4:
            return np.array(
                [best_quad[tl_idx], best_quad[tr_idx],
                 best_quad[bl_idx], best_quad[br_idx]],
                dtype=np.float32,
            )

        # Fallback ordering when indices collide
        order = np.argsort(sums)
        tl = best_quad[order[0]]
        br = best_quad[order[3]]
        remaining = best_quad[order[1:3]]
        if remaining[0][0] > remaining[1][0]:
            tr, bl = remaining[0], remaining[1]
        else:
            tr, bl = remaining[1], remaining[0]
        return np.array([tl, tr, bl, br], dtype=np.float32)

    def detect_court_surface(
        self, frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Detect the main court by its surface color and return 4 corners.

        Segments the court surface (blue/cyan), finds the largest quadrilateral
        contour, and returns its 4 corner points ordered as TL, TR, BL, BR.

        More robust than Hough-line-based detection in multi-court scenes
        because it isolates the single largest playing surface.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.surface_lower, self.surface_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < h * w * 0.03:
            return None

        peri = cv2.arcLength(largest, True)
        for eps_mult in [0.02, 0.03, 0.05, 0.08, 0.10]:
            approx = cv2.approxPolyDP(largest, eps_mult * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                sums = pts[:, 0] + pts[:, 1]
                diffs = pts[:, 1] - pts[:, 0]
                tl = pts[int(np.argmin(sums))]
                br = pts[int(np.argmax(sums))]
                tr = pts[int(np.argmin(diffs))]
                bl = pts[int(np.argmax(diffs))]
                return np.array([tl, tr, bl, br], dtype=np.float32)

        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect).astype(np.float32)
        sums = box[:, 0] + box[:, 1]
        diffs = box[:, 1] - box[:, 0]
        tl = box[int(np.argmin(sums))]
        br = box[int(np.argmax(sums))]
        tr = box[int(np.argmin(diffs))]
        bl = box[int(np.argmax(diffs))]
        return np.array([tl, tr, bl, br], dtype=np.float32)


# ============================================================================
# Segmentation-based Court Detector (YOLOv8-seg for Pickleball)
# ============================================================================

def extract_court_corners_from_segmentation(
    mask_or_polygon, frame_shape: Tuple
) -> Optional[np.ndarray]:
    """
    Extract 4 court corners from a segmentation polygon or mask.

    Reorders to TL, TR, BL, BR for homography correspondence with
    PICKLEBALL_COURT_CORNERS.
    """
    if isinstance(mask_or_polygon, np.ndarray) and mask_or_polygon.ndim == 2 and mask_or_polygon.shape[1] != 2:
        # Binary mask input
        contours, _ = cv2.findContours(
            mask_or_polygon.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        pts = None
        for eps in [0.02, 0.03, 0.05, 0.08]:
            approx = cv2.approxPolyDP(largest, eps * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                break
        if pts is None:
            pts = cv2.boxPoints(cv2.minAreaRect(largest)).astype(np.float32)
    else:
        # Polygon input (list of [x,y] points)
        poly = np.array(mask_or_polygon, dtype=np.float32)
        if len(poly) > 4:
            contour = poly.reshape(-1, 1, 2).astype(np.float32)
            peri = cv2.arcLength(contour, True)
            pts = None
            for eps in [0.02, 0.03, 0.05, 0.08, 0.10]:
                approx = cv2.approxPolyDP(contour, eps * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    break
            if pts is None:
                # Fallback: minimum area rectangle
                pts = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
        elif len(poly) == 4:
            pts = poly
        else:
            return None

    if pts is None or len(pts) < 4:
        return None

    pts = pts[:4]
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 1] - pts[:, 0]
    tl = pts[int(np.argmin(sums))]
    br = pts[int(np.argmax(sums))]
    tr = pts[int(np.argmin(diffs))]
    bl = pts[int(np.argmax(diffs))]

    return np.array([tl, tr, bl, br], dtype=np.float32)


def detect_court_lines_hybrid(
    frame: np.ndarray,
    court_mask: np.ndarray,
) -> Dict:
    """
    Detect precise court lines using hybrid approach:
    segmentation mask + classical CV (white line filtering + Hough)
    + dense contour keypoints + near baseline recovery.

    Args:
        frame: BGR frame
        court_mask: binary mask (0/1 uint8) of court region from seg model

    Returns:
        Dict with 'lines', 'baselines', 'sidelines', 'kitchen_lines',
        'keypoints', 'corners', 'dense_keypoints', 'contour_lines'
    """
    h, w = frame.shape[:2]
    result = {
        "lines": [],
        "baselines": [],
        "sidelines": [],
        "kitchen_lines": [],
        "keypoints": [],
        "corners": None,
        "dense_keypoints": [],
        "contour_lines": [],
        "center_line": None,
        "net_line": None,
    }

    # Step 1: White line detection within court mask
    # Erode court mask to avoid detecting seg boundary edges as court lines
    erode_kernel = np.ones((15, 15), np.uint8)
    court_mask_eroded = cv2.erode(court_mask, erode_kernel, iterations=1)

    # Lab L-channel: bright pixels in court region
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0]
    L_court = cv2.bitwise_and(L, L, mask=court_mask_eroded)
    court_pixels = L_court[court_mask_eroded > 0]
    if len(court_pixels) == 0:
        return result
    p90 = np.percentile(court_pixels, 90)
    white_lab = (L_court > p90).astype(np.uint8) * 255

    # HSV white: low saturation, high value
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_hsv = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))

    # Combine both masks, limit to eroded court
    white_mask = cv2.bitwise_or(white_lab, white_hsv)
    white_mask = cv2.bitwise_and(white_mask, white_mask, mask=court_mask_eroded)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    # Step 2: White detection on FULL court mask (for near baseline recovery)
    white_hsv_full = cv2.inRange(hsv, (0, 0, 140), (180, 90, 255))
    L_full = cv2.bitwise_and(L, L, mask=court_mask)
    cp_full = L_full[court_mask > 0]
    p88 = np.percentile(cp_full, 88) if len(cp_full) > 0 else 180
    white_lab_full = (L_full > p88).astype(np.uint8) * 255
    white_mask_full = cv2.bitwise_or(white_lab_full, white_hsv_full)
    white_mask_full = cv2.bitwise_and(white_mask_full, white_mask_full, mask=court_mask)
    white_mask_full = cv2.morphologyEx(white_mask_full, cv2.MORPH_OPEN, kernel)
    white_mask_full = cv2.morphologyEx(white_mask_full, cv2.MORPH_CLOSE, kernel)

    # Step 3: Dense contour keypoints from white regions
    contours, _ = cv2.findContours(
        white_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    dense_kps = []
    contour_lines_data = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, closed=False)
        if perimeter < 60 or len(cnt) < 5:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        aspect = max(rw, rh) / (min(rw, rh) + 1e-6)
        if aspect < 3:
            continue  # Not line-like, skip blobs
        if rw < rh:
            line_angle = 90 + angle
        else:
            line_angle = angle
        line_angle = abs(line_angle) % 180
        if line_angle < 25 or line_angle > 155:
            cls = "horizontal"
        elif 25 <= line_angle <= 75 or 105 <= line_angle <= 155:
            cls = "diagonal"
        else:
            cls = "vertical"
        contour_lines_data.append({
            "class": cls, "center": (int(cx), int(cy)),
            "angle": line_angle, "perimeter": perimeter,
        })
        n_samples = max(int(perimeter / 25), 3)
        for i in range(n_samples):
            idx = int(i * len(cnt) / n_samples) % len(cnt)
            dense_kps.append(tuple(cnt[idx][0]))
    result["dense_keypoints"] = dense_kps
    result["contour_lines"] = contour_lines_data

    # Step 4: Hough lines + validation
    edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)
    lines_p = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=80, maxLineGap=30,
    )

    def validate_line_on_white(x1, y1, x2, y2, white_img, min_ratio=0.4):
        n_samples = max(int(np.sqrt((x2-x1)**2 + (y2-y1)**2) / 3), 5)
        xs = np.clip(np.linspace(x1, x2, n_samples).astype(int), 0, white_img.shape[1]-1)
        ys = np.clip(np.linspace(y1, y2, n_samples).astype(int), 0, white_img.shape[0]-1)
        return sum(1 for x, y in zip(xs, ys) if white_img[y, x] > 0) / n_samples >= min_ratio

    valid_lines = []
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            if validate_line_on_white(x1, y1, x2, y2, white_mask):
                valid_lines.append(line)

    # Step 5: Classify & cluster Hough lines
    all_lines = []
    if valid_lines:
        for line in valid_lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
            length = np.sqrt(dx**2 + dy**2)
            all_lines.append({
                "pts": (x1, y1, x2, y2), "angle": angle,
                "length": length, "mid": ((x1+x2)/2, (y1+y2)/2),
                "slope": dy / (dx + 1e-6),
            })
    result["lines"] = all_lines

    horiz = [l for l in all_lines if l["angle"] < 20]
    sideline_cands = [l for l in all_lines if 25 <= l["angle"] <= 75]
    vertical_cands = [l for l in all_lines if l["angle"] > 75]  # Center line candidates

    # Step 4: Cluster lines
    def cluster_lines(lines_list, key_fn, threshold=25):
        if not lines_list:
            return []
        sorted_l = sorted(lines_list, key=key_fn)
        groups = [[sorted_l[0]]]
        for l in sorted_l[1:]:
            if abs(key_fn(l) - key_fn(groups[-1][-1])) < threshold:
                groups[-1].append(l)
            else:
                groups.append([l])
        # For each group, pick the longest line
        return [max(g, key=lambda x: x["length"]) for g in groups]

    h_clusters = cluster_lines(horiz, lambda l: l["mid"][1], threshold=25)

    # Step 5a: Classify horizontal lines with brightness validation
    # Strategy:
    # 1) Baselines = outermost lines → replaced by mask edges in step 6
    # 2) Net detected directly from white pixel row scan (not Hough)
    # 3) Kitchen = validated white court lines, NOT shadows or net fragments
    # 4) If far kitchen not detectable (perspective compression), infer from geometry
    net_line = None
    net_y = None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect net position by scanning white pixel rows in upper court
    # The net cable creates a narrow high-density white peak
    court_rows = np.where(np.sum(court_mask, axis=1) > 0)[0]
    if len(court_rows) > 0:
        court_top = court_rows[0]
        court_bot = court_rows[-1]
        search_end = int(court_top + (court_bot - court_top) * 0.15)
        best_net_y = None
        best_net_count = 0
        for scan_y in range(court_top + 5, search_end):
            row = white_mask_full[scan_y, :]
            n_white = int(np.sum(row > 0))
            if n_white > 200 and n_white > best_net_count:
                best_net_count = n_white
                best_net_y = scan_y
        if best_net_y is not None:
            cols = np.where(white_mask_full[best_net_y, :] > 0)[0]
            if len(cols) > 20:
                net_line = {"pts": (int(cols[0]), best_net_y, int(cols[-1]), best_net_y),
                            "angle": 0.0, "length": float(cols[-1] - cols[0]),
                            "mid": (float(np.mean(cols)), float(best_net_y)), "slope": 0.0}
                net_y = best_net_y

    if len(h_clusters) >= 2:
        h_sorted = sorted(h_clusters, key=lambda l: l["mid"][1])
        result["baselines"] = [h_sorted[0], h_sorted[-1]]

        if len(h_sorted) > 2:
            middle_lines = h_sorted[1:-1]

            # Kitchen lines: validate brightness + not near net
            kitchen_candidates = []
            for line in middle_lines:
                ly = int(line["mid"][1])
                # Skip if too close to net (within 60px = net artifact/shadow)
                if net_y is not None and abs(ly - net_y) < 60:
                    continue
                # Brightness check: average gray in center strip
                if 0 < ly < h:
                    cx_start = w // 4
                    cx_end = 3 * w // 4
                    strip = gray[ly, cx_start:cx_end]
                    avg_brightness = np.mean(strip)
                    # Court lines are white (brightness > 130)
                    if avg_brightness > 130:
                        kitchen_candidates.append(line)

            # Assign validated kitchen lines
            if len(kitchen_candidates) >= 2:
                kitchen_candidates.sort(key=lambda l: l["mid"][1])
                result["kitchen_lines"] = kitchen_candidates[:2]
            elif len(kitchen_candidates) == 1:
                result["kitchen_lines"] = kitchen_candidates
    elif len(h_clusters) == 1:
        result["baselines"] = h_clusters
    result["net_line"] = net_line

    # Step 5b: Detect center line (vertical, near court center x)
    center_line = None
    if vertical_cands:
        # Center line should be near the horizontal center of the court
        court_cols = np.where(np.sum(court_mask, axis=0) > 0)[0]
        if len(court_cols) > 0:
            court_center_x = (court_cols[0] + court_cols[-1]) / 2
            # Find longest vertical line near center
            center_cands = [l for l in vertical_cands
                           if abs(l["mid"][0] - court_center_x) < (court_cols[-1] - court_cols[0]) * 0.15]
            if center_cands:
                center_line = max(center_cands, key=lambda l: l["length"])
    result["center_line"] = center_line

    # Sidelines by slope direction (trapezoid: top < bottom)
    left_c, right_c = [], []
    for l in sideline_cands:
        x1, y1, x2, y2 = l["pts"]
        if y1 > y2: x1, y1, x2, y2 = x2, y2, x1, y1
        dx = x2 - x1
        if dx < -10: left_c.append(l)
        elif dx > 10: right_c.append(l)
    left_sl = max(left_c, key=lambda l: l["length"]) if left_c else None
    right_sl = max(right_c, key=lambda l: l["length"]) if right_c else None
    result["sidelines"] = [s for s in [left_sl, right_sl] if s]

    # Step 6a: Near baseline recovery — ALWAYS check for lower white lines
    if court_mask is not None:
        bot = white_mask_full[int(h*0.5):, :]
        row_counts = np.sum(bot > 0, axis=1)
        sig = np.where(row_counts > w * 0.05)[0]
        if len(sig) > 0:
            near_y = sig[-1] + int(h*0.5)
            cols = np.where(white_mask_full[near_y, :] > 0)[0]
            if len(cols) > 20:
                nb = {"pts": (int(cols[0]), near_y, int(cols[-1]), near_y),
                      "angle": 0.0, "length": float(cols[-1]-cols[0]),
                      "mid": (float(np.mean(cols)), float(near_y)), "slope": 0.0}
                cur_near_y = result["baselines"][-1]["mid"][1] if result["baselines"] else 0
                if near_y > cur_near_y + 30:
                    if len(result["baselines"]) >= 2:
                        # Just replace — don't demote to kitchen
                        result["baselines"][-1] = nb
                    elif len(result["baselines"]) == 1:
                        result["baselines"].append(nb)
                    else:
                        result["baselines"] = [nb]

    # Step 6b: Far baseline recovery from court mask top edge
    # The net often gets detected as far baseline — fix by finding
    # the actual top boundary of the court mask
    if court_mask is not None:
        court_rows = np.where(np.sum(court_mask, axis=1) > 10)[0]
        if len(court_rows) > 0:
            top_y = court_rows[0]
            # Scan a few rows near top for white pixels  
            for scan_y in range(top_y, min(top_y + 30, h)):
                cols = np.where(white_mask_full[scan_y, :] > 0)[0]
                if len(cols) > 15:
                    fb = {"pts": (int(cols[0]), scan_y, int(cols[-1]), scan_y),
                          "angle": 0.0, "length": float(cols[-1]-cols[0]),
                          "mid": (float(np.mean(cols)), float(scan_y)), "slope": 0.0}
                    cur_far_y = result["baselines"][0]["mid"][1] if result["baselines"] else h
                    if scan_y < cur_far_y - 20:
                        # Found baseline above current far baseline
                        if len(result["baselines"]) >= 2:
                            # Just replace — don't demote to kitchen
                            result["baselines"][0] = fb
                        elif len(result["baselines"]) == 1:
                            result["baselines"].insert(0, fb)
                        else:
                            result["baselines"] = [fb]
                    break
    if len(result["baselines"]) < 2 and court_mask is not None:
        top = white_mask_full[:int(h*0.5), :]
        row_counts = np.sum(top > 0, axis=1)
        sig = np.where(row_counts > w * 0.03)[0]
        if len(sig) > 0:
            far_y = sig[0]
            cols = np.where(white_mask_full[far_y, :] > 0)[0]
            if len(cols) > 20:
                fb = {"pts": (int(cols[0]), far_y, int(cols[-1]), far_y),
                      "angle": 0.0, "length": float(cols[-1]-cols[0]),
                      "mid": (float(np.mean(cols)), float(far_y)), "slope": 0.0}
                if not result["baselines"] or far_y < result["baselines"][0]["mid"][1]:
                    result["baselines"].insert(0, fb)
                else:
                    result["baselines"].append(fb)


    # Step 6c: Infer missing kitchen lines from court geometry
    # Pickleball: kitchen is 7ft from net, court half is 22ft
    # kitchen_ratio = 7/22 ≈ 0.318 from the net toward each baseline
    # In perspective, the far half is very compressed — far kitchen often
    # undetectable. Infer its position from baselines + net.
    if len(result["kitchen_lines"]) < 2 and len(result["baselines"]) >= 2:
        far_y = result["baselines"][0]["mid"][1]
        near_y = result["baselines"][-1]["mid"][1]
        far_x0 = result["baselines"][0]["pts"][0]
        far_x1 = result["baselines"][0]["pts"][2]
        near_x0 = result["baselines"][-1]["pts"][0]
        near_x1 = result["baselines"][-1]["pts"][2]

        # Estimate net position if not detected
        if net_y is None:
            # In perspective, net appears at ~35% from far baseline toward near
            net_y_est = far_y + (near_y - far_y) * 0.10
        else:
            net_y_est = net_y

        kitchen_ratio = 7.0 / 22.0  # 0.318

        existing_ys = set(int(k["mid"][1]) for k in result["kitchen_lines"])

        # Far kitchen: between far baseline and net
        far_kitchen_y = far_y + (net_y_est - far_y) * (1.0 - kitchen_ratio)
        # Near kitchen: between net and near baseline
        near_kitchen_y = net_y_est + (near_y - net_y_est) * kitchen_ratio

        def make_synthetic_line(y_pos):
            """Create a synthetic horizontal line at y_pos, interpolating x from baselines."""
            t = (y_pos - far_y) / max(near_y - far_y, 1)
            x0 = int(far_x0 + (near_x0 - far_x0) * t)
            x1 = int(far_x1 + (near_x1 - far_x1) * t)
            return {"pts": (x0, int(y_pos), x1, int(y_pos)),
                    "angle": 0.0, "length": float(x1 - x0),
                    "mid": (float((x0 + x1) / 2), float(y_pos)), "slope": 0.0}

        if len(result["kitchen_lines"]) == 0:
            # Both missing — add both
            result["kitchen_lines"] = [
                make_synthetic_line(far_kitchen_y),
                make_synthetic_line(near_kitchen_y),
            ]
        elif len(result["kitchen_lines"]) == 1:
            k_y = result["kitchen_lines"][0]["mid"][1]
            # Check if existing kitchen is near-side or far-side
            if net_y_est is not None and k_y > net_y_est:
                # Existing is near kitchen → add far kitchen
                result["kitchen_lines"].insert(0, make_synthetic_line(far_kitchen_y))
            else:
                # Existing is far kitchen → add near kitchen
                result["kitchen_lines"].append(make_synthetic_line(near_kitchen_y))
    # Step 7: Compute intersection keypoints
    def line_intersection(l1, l2):
        x1, y1, x2, y2 = l1["pts"]
        x3, y3, x4, y4 = l2["pts"]
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-6: return None
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        ix = x1 + t*(x2-x1)
        iy = y1 + t*(y2-y1)
        if -200 < ix < w+200 and -200 < iy < h+200:
            return (int(ix), int(iy))
        return None

    keypoints = []
    # Baseline × Sideline intersections (4 corners)
    for bl in result["baselines"]:
        for sl in result["sidelines"]:
            pt = line_intersection(bl, sl)
            if pt:
                keypoints.append(pt)
    # Kitchen × Sideline intersections
    for kl in result["kitchen_lines"]:
        for sl in result["sidelines"]:
            pt = line_intersection(kl, sl)
            if pt:
                keypoints.append(pt)
    # Baseline × Center line (service area dividers)
    if center_line:
        for bl in result["baselines"]:
            pt = line_intersection(bl, center_line)
            if pt:
                keypoints.append(pt)
        # Kitchen × Center line
        for kl in result["kitchen_lines"]:
            pt = line_intersection(kl, center_line)
            if pt:
                keypoints.append(pt)
    result["keypoints"] = keypoints

    # Step 8: Extract 4 corners + trapezoid validation
    if len(result["baselines"]) >= 2 and len(result["sidelines"]) >= 2:
        bl_s = sorted(result["baselines"], key=lambda l: l["mid"][1])
        corners = []
        for bl in bl_s:
            for sl in sorted(result["sidelines"], key=lambda l: l["mid"][0]):
                pt = line_intersection(bl, sl)
                if pt: corners.append(pt)
        if len(corners) >= 4:
            corners = np.array(corners[:4], dtype=np.float32)
            sums = corners[:, 0] + corners[:, 1]
            diffs = corners[:, 1] - corners[:, 0]
            tl = corners[int(np.argmin(sums))]
            br = corners[int(np.argmax(sums))]
            tr = corners[int(np.argmin(diffs))]
            bl_pt = corners[int(np.argmax(diffs))]
            ordered = np.array([tl, tr, bl_pt, br], dtype=np.float32)
            top_w = np.linalg.norm(tr - tl)
            bot_w = np.linalg.norm(br - bl_pt)
            if top_w < bot_w:
                result["corners"] = ordered

    return result


def draw_court_lines_overlay(
    frame: np.ndarray,
    court_lines: Dict,
    draw_keypoints: bool = True,
) -> np.ndarray:
    """
    Draw precise court lines overlay on frame.

    Args:
        frame: BGR frame to draw on
        court_lines: result from detect_court_lines_hybrid()
        draw_keypoints: whether to draw keypoint circles

    Returns:
        Annotated frame
    """
    out = frame.copy()
    corners = court_lines.get("corners")

    if corners is not None and len(corners) == 4:
        # Draw full court boundary using the 4 corners: TL, TR, BL, BR
        tl, tr, bl, br = [tuple(c.astype(int)) for c in corners]
        # Baselines (top and bottom) — green
        cv2.line(out, tl, tr, (0, 255, 0), 3)  # Far baseline
        cv2.line(out, bl, br, (0, 255, 0), 3)  # Near baseline
        # Sidelines (left and right) — green
        cv2.line(out, tl, bl, (0, 255, 0), 3)  # Left sideline
        cv2.line(out, tr, br, (0, 255, 0), 3)  # Right sideline
    else:
        # Fallback: draw raw detected lines
        for line in court_lines.get("baselines", []):
            x1, y1, x2, y2 = line["pts"]
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        for line in court_lines.get("sidelines", []):
            x1, y1, x2, y2 = line["pts"]
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Draw kitchen lines between their keypoint intersections (yellow)
    all_kps = court_lines.get("keypoints", [])
    n_baselines = len(court_lines.get("baselines", []))
    n_sidelines = len(court_lines.get("sidelines", []))
    n_kitchen = len(court_lines.get("kitchen_lines", []))
    n_corner_kps = n_baselines * n_sidelines  # Usually 4
    n_kitchen_sl_kps = n_kitchen * n_sidelines  # Kitchen × sidelines

    # Kitchen × sideline pairs (drawn as yellow lines)
    kitchen_sl_start = n_corner_kps
    kitchen_sl_end = kitchen_sl_start + n_kitchen_sl_kps
    for i in range(kitchen_sl_start, min(kitchen_sl_end, len(all_kps)) - 1, 2):
        pt1 = all_kps[i]
        pt2 = all_kps[i + 1]
        cv2.line(out, pt1, pt2, (0, 255, 255), 2)  # Yellow

    # Fallback: draw kitchen lines from raw Hough data
    if n_kitchen_sl_kps == 0 and court_lines.get("kitchen_lines"):
        for line in court_lines["kitchen_lines"]:
            x1, y1, x2, y2 = line["pts"]
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Draw center line (orange) if detected
    cl = court_lines.get("center_line")
    if cl:
        x1, y1, x2, y2 = cl["pts"]
        cv2.line(out, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange

    # Draw net line (white, thicker) if detected
    nl = court_lines.get("net_line")
    if nl:
        x1, y1, x2, y2 = nl["pts"]
        cv2.line(out, (x1, y1), (x2, y2), (255, 255, 255), 2)  # White

    # Draw dense keypoints along white line contours (cyan, small)
    for pt in court_lines.get("dense_keypoints", []):
        cv2.circle(out, pt, 4, (255, 255, 0), -1)  # Cyan filled
        cv2.circle(out, pt, 5, (255, 255, 255), 1)  # White outline

    # Draw intersection keypoints (red, larger — structural)
    if draw_keypoints:
        for pt in all_kps:
            cv2.circle(out, pt, 6, (0, 0, 255), -1)  # Red filled
            cv2.circle(out, pt, 7, (255, 255, 255), 2)  # White outline

    return out


class SegmentationCourtDetector:
    """
    Court detection using YOLOv8-seg for pickleball courts.

    Uses a trained segmentation model to detect court polygon, then extracts
    4 corners and computes homography to PICKLEBALL_COURT_CORNERS.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._model = None
        self.last_court_mask = None  # Cache for overlay drawing

    def _get_model(self):
        """Lazy-load YOLOv8-seg model."""
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            return self._model
        except Exception as e:
            raise RuntimeError(
                f"Cannot load pickleball court model {self.model_path}: {e}"
            ) from e

    def detect_and_compute_homography(
        self,
        frame: np.ndarray,
        court_keypoints: np.ndarray = PICKLEBALL_COURT_CORNERS,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Run segmentation, extract 4 corners, compute homography.

        Returns:
            (homography_matrix, result_dict)
        """
        result = {
            "method": "segmentation",
            "mask": None,
            "selected_corners": None,
            "homography": None,
        }
        if not os.path.exists(self.model_path):
            return None, result

        try:
            model = self._get_model()
        except RuntimeError:
            return None, result

        preds = model(frame, conf=self.conf_threshold, verbose=False)

        for r in preds:
            if r.masks is None:
                continue
            for j, (mask_data, box) in enumerate(zip(r.masks, r.boxes)):
                cls_id = int(box.cls[0])
                cls_name = r.names.get(cls_id, str(cls_id))
                if cls_name != "Court":
                    continue

                xy = mask_data.xy[0] if hasattr(mask_data, "xy") and len(mask_data.xy) > 0 else None

                # Cache the binary mask for hybrid line detection
                h_f, w_f = frame.shape[:2]
                mask_bin = mask_data.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_bin, (w_f, h_f))
                self.last_court_mask = (mask_resized > 0.5).astype(np.uint8)
                if xy is None:
                    continue

                corners = extract_court_corners_from_segmentation(xy, frame.shape)
                if corners is None:
                    continue

                H, _ = cv2.findHomography(
                    corners[:4].astype(np.float32),
                    court_keypoints[:4].astype(np.float32),
                    cv2.RANSAC, 5.0,
                )
                if H is None:
                    continue
                if ClassicalCourtDetector._validate_homography(
                    H, frame.shape, court_keypoints[:4]
                ):
                    result["selected_corners"] = corners
                    result["homography"] = H
                    return H, result

        return None, result


# ============================================================================
# SIFT-based Court Matching (for camera motion handling)
# ============================================================================

class SIFTCourtMatcher:
    """
    Uses SIFT feature matching to track court between frames,
    especially when the camera is moving (broadcast video).

    Computes relative homography between consecutive frames using
    SIFT keypoints + FLANN matcher + RANSAC.
    """

    def __init__(self):
        self.sift = cv2.SIFT_create()
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.prev_kp = None
        self.prev_desc = None
        self.cumulative_H = np.eye(3, dtype=np.float64)

    def compute_frame_homography(
        self, frame: np.ndarray, ratio_thresh: float = 0.7
    ) -> Optional[np.ndarray]:
        """
        Compute homography between current and previous frame using SIFT.

        Uses Lowe's ratio test for robust matching.

        Args:
            frame: Current BGR frame
            ratio_thresh: Lowe's ratio test threshold

        Returns:
            3x3 homography matrix from prev to current frame, or None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)

        if self.prev_desc is None:
            self.prev_kp = kp
            self.prev_desc = desc
            return np.eye(3)

        if desc is None or len(desc) < 4:
            return None

        matches = self.flann.knnMatch(self.prev_desc, desc, k=2)

        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            self.prev_kp = kp
            self.prev_desc = desc
            return None

        src_pts = np.float32(
            [self.prev_kp[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        self.prev_kp = kp
        self.prev_desc = desc

        if H is not None:
            self.cumulative_H = H @ self.cumulative_H

        return H


# ============================================================================
# Deep Learning Court Keypoint Detector
# ============================================================================

class CourtKeypointNet:
    """
    CNN-based court keypoint detector.

    Architecture: ResNet-18 backbone + regression head.
    Predicts (x, y) coordinates for N court keypoints.

    Input: RGB image (3, H, W)
    Output: (N, 2) keypoint coordinates normalized to [0, 1]
    """

    def __init__(self, num_keypoints: int = 14, pretrained: bool = True):
        _lazy_import_torch()
        if not HAS_TORCH or not HAS_TORCHVISION:
            raise RuntimeError(
                "CourtKeypointNet requires torch and torchvision. "
                "Install them: pip install torch torchvision"
            )
        # Dynamically set base class to nn.Module
        self.__class__.__bases__ = (nn.Module,)
        nn.Module.__init__(self)
        self.num_keypoints = num_keypoints

        # Backbone: ResNet-18 pretrained on ImageNet
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Remove the final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Regression head: predict (x, y) for each keypoint
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_keypoints * 2),
            nn.Sigmoid(),  # Output normalized [0, 1]
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, 3, H, W) input tensor
        Returns:
            (B, num_keypoints, 2) predicted keypoint coordinates
        """
        features = self.features(x)  # (B, 512, 1, 1)
        coords = self.regressor(features)  # (B, num_keypoints * 2)
        return coords.view(-1, self.num_keypoints, 2)


class DeepCourtDetector:
    """
    Deep learning-based court detector using CourtKeypointNet.

    Predicts 14 court keypoints directly from the image,
    then computes homography from the predicted keypoints.
    """

    def __init__(
        self,
        num_keypoints: int = 14,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        input_size: Tuple[int, int] = (640, 360),
    ):
        _lazy_import_torch()
        if not HAS_TORCH:
            raise RuntimeError("DeepCourtDetector requires torch.")
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.input_size = input_size  # (width, height)
        self.num_keypoints = num_keypoints

        self.model = CourtKeypointNet(
            num_keypoints=num_keypoints, pretrained=True
        ).to(self.device)

        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)

        self.model.eval()

    def preprocess(self, frame: np.ndarray):
        """Resize and normalize frame for the network."""
        resized = cv2.resize(frame, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device)

    def detect_keypoints(
        self, frame: np.ndarray
    ) -> np.ndarray:
        """
        Predict court keypoints from a frame.

        Args:
            frame: BGR image

        Returns:
            (N, 2) array of keypoint pixel coordinates in original frame
        """
        h, w = frame.shape[:2]
        tensor = self.preprocess(frame)
        with torch.no_grad():
            pred = self.model(tensor)  # (1, N, 2) normalized coords
        keypoints = pred[0].cpu().numpy()  # (N, 2)

        # Scale back to original frame size
        keypoints[:, 0] *= w
        keypoints[:, 1] *= h

        return keypoints

    def detect_and_compute_homography(
        self,
        frame: np.ndarray,
        court_keypoints: np.ndarray = TENNIS_COURT_KEYPOINTS[:14],
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Detect keypoints and compute homography.

        Returns:
            (H, predicted_keypoints)
        """
        image_keypoints = self.detect_keypoints(frame)

        if len(image_keypoints) >= 4 and len(court_keypoints) >= 4:
            n = min(len(image_keypoints), len(court_keypoints))
            H, mask = cv2.findHomography(
                image_keypoints[:n].astype(np.float32),
                court_keypoints[:n].astype(np.float32),
                cv2.RANSAC,
                5.0,
            )
            return H, image_keypoints

        return None, image_keypoints


# ============================================================================
# Utility Functions
# ============================================================================

def transform_point(
    H: np.ndarray, point: np.ndarray
) -> np.ndarray:
    """
    Transform a 2D point using homography matrix.

    Args:
        H: 3x3 homography matrix
        point: (x, y) pixel coordinate

    Returns:
        (x', y') transformed coordinate
    """
    p = np.array([point[0], point[1], 1.0])
    tp = H @ p
    if abs(tp[2]) > 1e-8:
        return tp[:2] / tp[2]
    return tp[:2]


def transform_points(
    H: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """
    Transform multiple 2D points using homography matrix.

    Args:
        H: 3x3 homography matrix
        points: (N, 2) array of pixel coordinates

    Returns:
        (N, 2) array of transformed coordinates
    """
    if len(points) == 0:
        return np.empty((0, 2))

    pts = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float32), H
    )
    return pts.reshape(-1, 2)


def draw_court_overlay(
    frame: np.ndarray,
    H_inv: np.ndarray,
    court_keypoints: np.ndarray = TENNIS_COURT_CORNERS,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw court lines on the video frame using the inverse homography.

    Projects court model lines back into image space.
    """
    overlay = frame.copy()

    # Project court corners to image
    image_pts = transform_points(H_inv, court_keypoints)

    # Draw lines connecting court corners
    n = len(image_pts)
    if n >= 4:
        pts_int = image_pts.astype(int)
        # Draw outline rectangle
        cv2.line(overlay, tuple(pts_int[0]), tuple(pts_int[1]), color, thickness)
        cv2.line(overlay, tuple(pts_int[1]), tuple(pts_int[3]), color, thickness)
        cv2.line(overlay, tuple(pts_int[3]), tuple(pts_int[2]), color, thickness)
        cv2.line(overlay, tuple(pts_int[2]), tuple(pts_int[0]), color, thickness)

    return overlay


class CourtTracker:
    """Temporal smoothing for court detection across video frames.

    Provides:
    - Reference frame lock: first N frames establish stable reference
    - EMA smoothing: reduces jitter between consecutive frames
    - Outlier rejection: keypoints that jump too far are replaced
    - Confidence fallback: reuse last good detection on bad frames
    """

    def __init__(self, alpha: float = 0.3, outlier_px: float = 50.0,
                 ref_frames: int = 10):
        self.alpha = alpha              # EMA smoothing factor (0=all prev, 1=all new)
        self.outlier_px = outlier_px    # Max allowed keypoint movement from reference
        self.ref_frames = ref_frames    # Frames to collect before locking reference
        self.ref_keypoints = None       # Locked reference keypoints (median of first N)
        self.prev_keypoints = None      # Previous frame's smoothed keypoints
        self.prev_court_lines = None    # Previous frame's full court_lines dict
        self.history = []               # Keypoint buffer for reference computation
        self.frames_since_reference = 0
        self.rejection_count = 0        # Consecutive frames with high rejection

    def update(self, court_lines: Dict) -> Dict:
        """Smooth court_lines keypoints using temporal tracking.

        Args:
            court_lines: result from detect_court_lines_hybrid()

        Returns:
            Smoothed court_lines dict
        """
        kps = court_lines.get("keypoints", [])

        # No keypoints detected — fallback to previous
        if not kps:
            if self.prev_court_lines is not None:
                return self.prev_court_lines
            return court_lines

        kps_arr = np.array(kps, dtype=np.float64)

        # Phase 1: Collecting reference frames
        if len(self.history) < self.ref_frames:
            self.history.append(kps_arr.copy())
            self.prev_keypoints = kps_arr
            self.prev_court_lines = court_lines
            # Lock reference once we have enough frames
            if len(self.history) == self.ref_frames:
                self._build_reference()
            return court_lines

        # Phase 2: Temporal smoothing
        n_kps = min(len(kps_arr), len(self.prev_keypoints))
        if n_kps == 0:
            if self.prev_court_lines is not None:
                return self.prev_court_lines
            return court_lines

        smoothed = kps_arr[:n_kps].copy()
        n_rejected = 0

        for i in range(n_kps):
            new_pt = kps_arr[i]
            prev_pt = self.prev_keypoints[i] if i < len(self.prev_keypoints) else new_pt

            # EMA smoothing
            smoothed_pt = self.alpha * new_pt + (1 - self.alpha) * prev_pt

            # Outlier rejection against reference
            if self.ref_keypoints is not None and i < len(self.ref_keypoints):
                ref_pt = self.ref_keypoints[i]
                dist = np.linalg.norm(smoothed_pt - ref_pt)
                if dist > self.outlier_px:
                    # Too far from reference — use reference instead
                    smoothed_pt = ref_pt
                    n_rejected += 1

            smoothed[i] = smoothed_pt

        # Convert back to tuple list
        smoothed_kps = [tuple(pt.astype(int)) for pt in smoothed]
        court_lines["keypoints"] = smoothed_kps
        self.prev_keypoints = smoothed.copy()

        # Update corners from smoothed keypoints (first 4 are corners)
        if len(smoothed_kps) >= 4 and court_lines.get("corners") is not None:
            court_lines["corners"] = np.array(smoothed_kps[:4], dtype=np.float32)

        # Re-detection trigger: if >70% keypoints rejected, rebuild reference
        rejection_rate = n_rejected / max(n_kps, 1)
        if rejection_rate > 0.7:
            self.rejection_count += 1
            if self.rejection_count > 15:  # 15 consecutive bad frames
                self.history = []
                self.ref_keypoints = None
                self.rejection_count = 0
        else:
            self.rejection_count = 0

        self.prev_court_lines = court_lines
        self.frames_since_reference += 1
        return court_lines

    def _build_reference(self):
        """Compute reference keypoints as median of collected frames."""
        if not self.history:
            return
        # Pad shorter arrays to match the longest
        max_len = max(len(h) for h in self.history)
        padded = []
        for h in self.history:
            if len(h) < max_len:
                pad = np.zeros((max_len - len(h), 2))
                padded.append(np.vstack([h, pad]))
            else:
                padded.append(h[:max_len])
        stacked = np.stack(padded, axis=0)  # (N_frames, N_kps, 2)
        self.ref_keypoints = np.median(stacked, axis=0)
        self.prev_keypoints = self.ref_keypoints.copy()
