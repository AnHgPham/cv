"""
Module 3: Object Tracking
=========================

Implements three tracking approaches:

1. Kalman Filter - Ball Tracking:
   Linear state estimation with constant velocity model.
   State: [x, y, vx, vy], predicts ball position when detection is missing.

2. Optical Flow (Lucas-Kanade) - Motion Estimation:
   Sparse optical flow to estimate instantaneous velocity of the ball
   between consecutive frames. Feeds into Kalman Filter.

3. DeepSORT - Player Tracking:
   Combines Kalman Filter prediction with deep appearance features
   for robust multi-object tracking with ID assignment.

Knowledge applied:
- Kalman Filter (linear state estimation, predict/update cycle)
- Optical Flow (Lucas-Kanade pyramidal method)
- DeepSORT (Kalman + Hungarian assignment + appearance Re-ID)
"""

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

# Import Detection from our object_detection module
try:
    from .object_detection import Detection
except ImportError:
    from object_detection import Detection


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Track:
    """Represents a tracked object over time."""
    track_id: int
    class_name: str
    positions: List[Tuple[float, float]] = field(default_factory=list)
    velocities: List[Tuple[float, float]] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    is_active: bool = True
    frames_since_detection: int = 0
    total_detections: int = 0

    @property
    def last_position(self) -> Optional[Tuple[float, float]]:
        return self.positions[-1] if self.positions else None

    @property
    def last_velocity(self) -> Optional[Tuple[float, float]]:
        return self.velocities[-1] if self.velocities else None


# ============================================================================
# 1. Kalman Filter - Ball Tracking
# ============================================================================

class BallKalmanTracker:
    """
    Kalman Filter-based ball tracker.

    State model (Constant Velocity):
        State vector x = [x, y, vx, vy]^T
        Transition:   x_{k+1} = F * x_k + w   (process noise)
        Measurement:  z_k = H * x_k + v        (measurement noise)

    Where:
        F = [[1, 0, dt, 0],    (position += velocity * dt)
             [0, 1, 0, dt],
             [0, 0, 1,  0],    (velocity stays constant)
             [0, 0, 0,  1]]

        H = [[1, 0, 0, 0],    (we observe position only)
             [0, 1, 0, 0]]

    The Kalman Filter maintains:
    - Predicted state (where we think the ball is)
    - State covariance (uncertainty in our estimate)

    Two-step cycle every frame:
    1. PREDICT: Propagate state forward using motion model
    2. UPDATE: Correct prediction using new measurement (if available)
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise_std: float = 5.0,
        measurement_noise_std: float = 2.0,
        initial_covariance: float = 100.0,
        max_missing_frames: int = 10,
    ):
        """
        Args:
            dt: Time step (1.0 for frame-by-frame)
            process_noise_std: Std dev of process noise (acceleration uncertainty)
            measurement_noise_std: Std dev of measurement noise (detection uncertainty)
            initial_covariance: Initial state covariance (uncertainty at start)
            max_missing_frames: Kill track after this many frames without detection
        """
        self.dt = dt
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.initial_covariance = initial_covariance
        self.max_missing_frames = max_missing_frames

        self.kf: Optional[KalmanFilter] = None
        self.track: Optional[Track] = None
        self.track_id_counter = 0
        self.frame_count = 0

    def _create_kalman_filter(self, x0: float, y0: float) -> KalmanFilter:
        """
        Initialize a new Kalman Filter with the first detection.

        Sets up the state transition matrix F, measurement matrix H,
        process noise Q, measurement noise R, and initial state.
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State Transition Matrix F (constant velocity)
        kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement Matrix H (observe position only)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process Noise Covariance Q
        q = self.process_noise_std ** 2
        kf.Q = np.array([
            [q * self.dt**4 / 4, 0, q * self.dt**3 / 2, 0],
            [0, q * self.dt**4 / 4, 0, q * self.dt**3 / 2],
            [q * self.dt**3 / 2, 0, q * self.dt**2, 0],
            [0, q * self.dt**3 / 2, 0, q * self.dt**2],
        ], dtype=np.float64)

        # Measurement Noise Covariance R
        r = self.measurement_noise_std ** 2
        kf.R = np.array([
            [r, 0],
            [0, r],
        ], dtype=np.float64)

        # Initial State Covariance P
        kf.P *= self.initial_covariance

        # Initial State
        kf.x = np.array([[x0], [y0], [0], [0]], dtype=np.float64)

        return kf

    def predict(self) -> Tuple[float, float]:
        """
        Kalman PREDICT step: propagate state forward.

        Returns predicted (x, y) position.
        """
        if self.kf is None:
            return (0.0, 0.0)

        self.kf.predict()
        return (float(self.kf.x[0, 0]), float(self.kf.x[1, 0]))

    def update(self, detection: Optional[Detection]) -> Tuple[float, float]:
        """
        Kalman UPDATE step: correct prediction with new measurement.

        If detection is None, only prediction is used (no correction).

        Args:
            detection: Ball detection for current frame, or None

        Returns:
            Estimated (x, y) position after update
        """
        self.frame_count += 1

        if detection is not None:
            cx, cy = detection.center

            if self.kf is None:
                # First detection: initialize filter
                self.kf = self._create_kalman_filter(cx, cy)
                self.track_id_counter += 1
                self.track = Track(
                    track_id=self.track_id_counter,
                    class_name="ball",
                )

            # Predict then update
            self.kf.predict()
            self.kf.update(np.array([[cx], [cy]]))

            self.track.frames_since_detection = 0
            self.track.total_detections += 1

        else:
            if self.kf is not None:
                # No detection: predict only
                self.kf.predict()
                self.track.frames_since_detection += 1

                # Kill track if missing too long
                if self.track.frames_since_detection > self.max_missing_frames:
                    self.track.is_active = False

        if self.kf is not None and self.track is not None:
            pos = (float(self.kf.x[0, 0]), float(self.kf.x[1, 0]))
            vel = (float(self.kf.x[2, 0]), float(self.kf.x[3, 0]))
            self.track.positions.append(pos)
            self.track.velocities.append(vel)
            self.track.frames.append(self.frame_count)
            conf = detection.confidence if detection else 0.0
            self.track.confidences.append(conf)
            return pos

        return (0.0, 0.0)

    def get_state(self) -> Optional[np.ndarray]:
        """Get current Kalman state [x, y, vx, vy]."""
        if self.kf is not None:
            return self.kf.x.flatten()
        return None

    def get_covariance(self) -> Optional[np.ndarray]:
        """Get current state covariance matrix P."""
        if self.kf is not None:
            return self.kf.P.copy()
        return None

    def reset(self):
        """Reset the tracker to initial state."""
        self.kf = None
        self.track = None
        self.frame_count = 0


# ============================================================================
# 2. Optical Flow - Lucas-Kanade
# ============================================================================

class OpticalFlowEstimator:
    """
    Sparse Optical Flow using Lucas-Kanade pyramidal method.

    Computes motion vectors for specific points (ball, players)
    between consecutive frames. Used to:
    1. Estimate instantaneous velocity of the ball
    2. Supplement Kalman Filter predictions
    3. Handle cases where detection temporarily fails

    Lucas-Kanade Assumptions:
    - Brightness constancy: I(x,y,t) = I(x+dx, y+dy, t+dt)
    - Small motion: Taylor expansion is valid
    - Spatial coherence: nearby pixels have similar flow

    The method solves the optical flow equation:
        Ix*u + Iy*v + It = 0
    for velocity (u, v) in a local window around each point.
    """

    def __init__(
        self,
        win_size: Tuple[int, int] = (21, 21),
        max_level: int = 3,
        criteria_max_count: int = 30,
        criteria_epsilon: float = 0.01,
    ):
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                criteria_max_count,
                criteria_epsilon,
            ),
        )
        self.prev_gray: Optional[np.ndarray] = None

    def compute_flow(
        self,
        frame: np.ndarray,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute optical flow for given points between previous and current frame.

        Args:
            frame: Current BGR frame
            points: (N, 2) array of points to track

        Returns:
            new_points: (N, 2) tracked point positions in current frame
            velocities: (N, 2) velocity vectors (displacement per frame)
            status: (N,) boolean array, True if tracking succeeded
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or len(points) == 0:
            self.prev_gray = gray
            return points, np.zeros_like(points), np.ones(len(points), dtype=bool)

        # Format points for OpenCV
        pts = points.reshape(-1, 1, 2).astype(np.float32)

        # Lucas-Kanade optical flow
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, pts, None, **self.lk_params
        )

        self.prev_gray = gray

        if new_pts is None:
            return points, np.zeros_like(points), np.zeros(len(points), dtype=bool)

        new_pts = new_pts.reshape(-1, 2)
        status = status.flatten().astype(bool)
        velocities = new_pts - points

        return new_pts, velocities, status

    def estimate_ball_velocity(
        self,
        frame: np.ndarray,
        ball_position: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate the ball's velocity using optical flow.

        Computes flow for the ball's position and nearby points,
        returning the median velocity to be robust to noise.

        Args:
            frame: Current BGR frame
            ball_position: (x, y) ball center

        Returns:
            (vx, vy) velocity vector, or None if flow computation fails
        """
        cx, cy = ball_position

        # Create a small grid of points around the ball
        offsets = np.array([
            [0, 0], [-2, -2], [2, -2], [-2, 2], [2, 2],
            [-1, 0], [1, 0], [0, -1], [0, 1],
        ], dtype=np.float32)

        points = np.array([[cx, cy]], dtype=np.float32) + offsets

        new_points, velocities, status = self.compute_flow(frame, points)

        if not np.any(status):
            return None

        # Median velocity of tracked points (robust to outliers)
        valid_vel = velocities[status]
        median_vel = np.median(valid_vel, axis=0)

        return (float(median_vel[0]), float(median_vel[1]))

    def reset(self):
        """Reset the optical flow estimator."""
        self.prev_gray = None


# ============================================================================
# 3. DeepSORT - Player Tracking
# ============================================================================

class DeepSORTTracker:
    """
    DeepSORT-based multi-object tracker for player tracking.

    Extends SORT (Simple Online and Realtime Tracking) with:
    - Deep appearance features (Re-ID) for robust association
    - Kalman Filter for motion prediction
    - Hungarian algorithm for detection-to-track assignment

    Pipeline per frame:
    1. Receive player detections from object detector
    2. Predict track positions using Kalman Filter
    3. Compute cost matrix (IoU + appearance distance)
    4. Solve assignment problem (Hungarian algorithm)
    5. Update matched tracks, create new tracks, delete lost tracks

    Uses the 'deep-sort-realtime' library for implementation.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.3,
        nn_budget: int = 100,
        embedder: str = "mobilenet",
    ):
        """
        Args:
            max_age: Max frames to keep track alive without detection
            n_init: Min detections before track is confirmed
            max_iou_distance: Max IoU distance for matching
            max_cosine_distance: Max cosine distance for appearance
            nn_budget: Max gallery size per track
            embedder: Appearance feature extractor model
        """
        self.max_age = max_age
        self.n_init = n_init
        self.tracks: Dict[int, Track] = {}
        self.frame_count = 0

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance,
                max_cosine_distance=max_cosine_distance,
                nn_budget=nn_budget,
                embedder=embedder,
            )
            self._use_deepsort = True
        except ImportError:
            print(
                "WARNING: deep-sort-realtime not installed. "
                "Falling back to simple IoU-based tracking."
            )
            self._use_deepsort = False
            self._simple_tracks: List[Dict] = []
            self._next_id = 1

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> List[Tuple[int, Detection]]:
        """
        Update tracker with new detections.

        Args:
            detections: List of player detections for current frame
            frame: Current BGR frame (needed for appearance features)

        Returns:
            List of (track_id, detection) tuples for active tracks
        """
        self.frame_count += 1

        if self._use_deepsort:
            return self._update_deepsort(detections, frame)
        else:
            return self._update_simple_iou(detections)

    def _update_deepsort(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> List[Tuple[int, Detection]]:
        """Update using DeepSORT library."""
        # Convert detections to format expected by deep-sort-realtime
        # Format: [[x1, y1, w, h, confidence], ...]
        det_list = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            det_list.append(([x1, y1, w, h], det.confidence, det.class_name))

        # Update tracker
        tracks = self.tracker.update_tracks(det_list, frame=frame)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]

            det = Detection(
                bbox=(int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])),
                confidence=track.det_conf if track.det_conf else 0.5,
                class_id=1,
                class_name="player",
            )

            # Update internal track history
            if track_id not in self.tracks:
                self.tracks[track_id] = Track(
                    track_id=track_id, class_name="player"
                )
            self.tracks[track_id].positions.append(det.center)
            self.tracks[track_id].frames.append(self.frame_count)

            results.append((track_id, det))

        return results

    def _update_simple_iou(
        self, detections: List[Detection]
    ) -> List[Tuple[int, Detection]]:
        """Simple IoU-based tracking fallback (no appearance features)."""
        results = []

        if not self._simple_tracks:
            # Initialize tracks from first detections
            for det in detections:
                track = {
                    "id": self._next_id,
                    "bbox": det.bbox,
                    "age": 0,
                    "hits": 1,
                }
                self._simple_tracks.append(track)
                self._next_id += 1
                results.append((track["id"], det))
            return results

        # Compute IoU cost matrix
        from .object_detection import _compute_iou

        cost_matrix = np.zeros(
            (len(self._simple_tracks), len(detections))
        )
        for i, track in enumerate(self._simple_tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = _compute_iou(track["bbox"], det.bbox)

        # Greedy assignment (simplified Hungarian)
        matched_tracks = set()
        matched_dets = set()

        while True:
            if cost_matrix.size == 0:
                break
            max_val = np.max(cost_matrix)
            if max_val < 0.3:
                break
            idx = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
            i, j = idx
            if i in matched_tracks or j in matched_dets:
                cost_matrix[i, j] = 0
                continue

            self._simple_tracks[i]["bbox"] = detections[j].bbox
            self._simple_tracks[i]["age"] = 0
            self._simple_tracks[i]["hits"] += 1
            results.append((self._simple_tracks[i]["id"], detections[j]))
            matched_tracks.add(i)
            matched_dets.add(j)
            cost_matrix[i, :] = 0
            cost_matrix[:, j] = 0

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                track = {
                    "id": self._next_id,
                    "bbox": det.bbox,
                    "age": 0,
                    "hits": 1,
                }
                self._simple_tracks.append(track)
                self._next_id += 1
                results.append((track["id"], det))

        # Age unmatched tracks and remove old ones
        for i in range(len(self._simple_tracks) - 1, -1, -1):
            if i not in matched_tracks:
                self._simple_tracks[i]["age"] += 1
                if self._simple_tracks[i]["age"] > self.max_age:
                    self._simple_tracks.pop(i)

        return results

    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track history by ID."""
        return self.tracks.get(track_id)

    def get_all_active_tracks(self) -> List[Track]:
        """Get all currently active tracks."""
        return [t for t in self.tracks.values() if t.is_active]


# ============================================================================
# Combined Tracker: Kalman + Optical Flow for Ball
# ============================================================================

class BallTracker:
    """
    Combined ball tracker using Kalman Filter + Optical Flow.

    Workflow per frame:
    1. Receive ball detection (or None if not detected)
    2. Compute optical flow velocity estimate
    3. If detection available: Kalman update with measurement
    4. If no detection: Use optical flow velocity to improve Kalman prediction
    5. Return smoothed position estimate

    This fusion improves robustness when the ball is temporarily
    occluded or missed by the detector.
    """

    def __init__(
        self,
        process_noise_std: float = 5.0,
        measurement_noise_std: float = 2.0,
        max_missing_frames: int = 10,
    ):
        self.kalman = BallKalmanTracker(
            process_noise_std=process_noise_std,
            measurement_noise_std=measurement_noise_std,
            max_missing_frames=max_missing_frames,
        )
        self.optical_flow = OpticalFlowEstimator()
        self.last_position: Optional[Tuple[float, float]] = None

    def update(
        self,
        frame: np.ndarray,
        detection: Optional[Detection],
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Track]]:
        """
        Process a new frame with optional ball detection.

        Args:
            frame: Current BGR frame
            detection: Ball detection (or None)

        Returns:
            (estimated_position, track_object) or (None, None)
        """
        # Compute optical flow if we have a previous position
        of_velocity = None
        if self.last_position is not None:
            of_velocity = self.optical_flow.estimate_ball_velocity(
                frame, self.last_position
            )

        # Kalman update
        position = self.kalman.update(detection)

        # If no detection but optical flow available, we can
        # use it to adjust the Kalman state velocity
        if detection is None and of_velocity is not None and self.kalman.kf is not None:
            # Blend optical flow velocity with Kalman velocity
            kf_vx = self.kalman.kf.x[2, 0]
            kf_vy = self.kalman.kf.x[3, 0]
            of_vx, of_vy = of_velocity

            # Weighted average (trust Kalman more when track is established)
            alpha = 0.3  # optical flow weight
            self.kalman.kf.x[2, 0] = (1 - alpha) * kf_vx + alpha * of_vx
            self.kalman.kf.x[3, 0] = (1 - alpha) * kf_vy + alpha * of_vy

        if self.kalman.track and self.kalman.track.is_active:
            self.last_position = position
            return position, self.kalman.track
        else:
            self.last_position = None
            return None, None

    def reset(self):
        """Reset the combined tracker."""
        self.kalman.reset()
        self.optical_flow.reset()
        self.last_position = None
