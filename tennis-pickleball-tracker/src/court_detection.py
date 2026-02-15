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
        lines: List[np.ndarray], distance_thresh: float = 30.0
    ) -> List[np.ndarray]:
        """
        Merge lines that are close together (likely the same court line).

        Groups lines by proximity and averages each group into one line.
        """
        if not lines:
            return []

        lines = sorted(lines, key=lambda l: (l[1] + l[3]) / 2)
        merged = []
        current_group = [lines[0]]

        for i in range(1, len(lines)):
            mid_prev = (current_group[-1][1] + current_group[-1][3]) / 2
            mid_curr = (lines[i][1] + lines[i][3]) / 2

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
        mask = self.preprocess(frame)
        result["mask"] = mask

        # Step 4: Edge detection
        edges = self.detect_edges(mask)
        result["edges"] = edges

        # Step 5: Hough lines
        raw_lines = self.detect_lines(edges)
        if raw_lines is None or len(raw_lines) == 0:
            return result
        result["lines"] = raw_lines

        # Step 6: Classify lines
        horizontal, vertical = self.classify_lines(raw_lines)

        # Merge similar lines
        horizontal = self.merge_similar_lines(horizontal, distance_thresh=30)
        vertical = self.merge_similar_lines(vertical, distance_thresh=30)
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

        Automatically selects the best 4 corner points from intersections
        and maps them to the standard court corners.

        Returns:
            (homography_matrix, detection_result_dict)
        """
        result = self.detect(frame)
        intersections = result["intersections"]

        if len(intersections) < 4:
            return None, result

        # Select 4 corner-like points (top-left, top-right, bottom-left, bottom-right)
        pts = np.array(intersections, dtype=np.float32)
        corners = self._select_court_corners(pts, frame.shape)

        if corners is not None and len(corners) >= 4:
            H = self.compute_homography(corners[:4], court_keypoints[:4])
            result["homography"] = H
            result["selected_corners"] = corners
            return H, result

        return None, result

    @staticmethod
    def _select_court_corners(
        points: np.ndarray, frame_shape: Tuple
    ) -> Optional[np.ndarray]:
        """
        Select 4 court corner points from candidate intersections.

        Uses centroid-based sorting: compute centroid, then classify
        points into quadrants (TL, TR, BL, BR).
        """
        if len(points) < 4:
            return None

        h, w = frame_shape[:2]
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])

        tl_candidates = points[(points[:, 0] < cx) & (points[:, 1] < cy)]
        tr_candidates = points[(points[:, 0] >= cx) & (points[:, 1] < cy)]
        bl_candidates = points[(points[:, 0] < cx) & (points[:, 1] >= cy)]
        br_candidates = points[(points[:, 0] >= cx) & (points[:, 1] >= cy)]

        corners = []
        for candidates in [tl_candidates, tr_candidates, bl_candidates, br_candidates]:
            if len(candidates) == 0:
                return None
            # Pick the point closest to the respective image corner
            corners.append(candidates[0])

        return np.array(corners, dtype=np.float32)


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
