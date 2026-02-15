"""
Module 4: 3D Trajectory Reconstruction
=======================================

Reconstructs the 3D trajectory of the ball from 2D video observations
using a single camera, homography, and physics constraints.

Steps:
1. 2D -> Court Projection: Use homography H to map ball pixel coords
   to real-world (x, y) on the court surface
2. Height Estimation: Use physics model (parabolic trajectory under gravity)
   to estimate the z (height) component
3. Extended Kalman Filter: 6D state [x, y, z, vx, vy, vz] with nonlinear
   observation model
4. Bounce Detection: Detect when ball hits the ground using trajectory features
5. In/Out Classification: Compare bounce point to court boundaries

Knowledge applied:
- Homography (2D -> court coordinate projection)
- Extended Kalman Filter (nonlinear state estimation)
- Physics modeling (projectile motion with gravity)
- Machine Learning (bounce classification)
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from scipy.optimize import curve_fit

try:
    from .court_detection import transform_point, transform_points
    from .court_detection import TENNIS_COURT_KEYPOINTS, TENNIS_COURT_CORNERS
except ImportError:
    from court_detection import transform_point, transform_points
    from court_detection import TENNIS_COURT_KEYPOINTS, TENNIS_COURT_CORNERS


# ============================================================================
# Constants
# ============================================================================

GRAVITY = 9.81  # m/s^2
TENNIS_BALL_RADIUS = 0.033  # meters
TENNIS_BALL_MASS = 0.057  # kg
DRAG_COEFFICIENT = 0.55  # approximate air drag coefficient for tennis ball
AIR_DENSITY = 1.225  # kg/m^3
BALL_CROSS_SECTION = np.pi * TENNIS_BALL_RADIUS ** 2  # m^2

# Court boundaries for in/out decision (tennis singles)
TENNIS_SINGLES_BOUNDS = {
    "x_min": 0.0,
    "x_max": 23.77,
    "y_min": 1.37,   # singles sideline offset from doubles
    "y_max": 9.60,
}

TENNIS_DOUBLES_BOUNDS = {
    "x_min": 0.0,
    "x_max": 23.77,
    "y_min": 0.0,
    "y_max": 10.97,
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TrajectoryPoint3D:
    """A single 3D point in the ball trajectory."""
    frame_id: int
    x: float  # court x coordinate (meters)
    y: float  # court y coordinate (meters)
    z: float  # height above court (meters)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    is_bounce: bool = False
    is_estimated: bool = False  # True if z was estimated from physics


@dataclass
class BounceEvent:
    """Detected bounce event."""
    frame_id: int
    x: float  # court x coordinate
    y: float  # court y coordinate
    is_in: bool  # True if ball is in bounds
    confidence: float = 0.0


@dataclass
class Trajectory3D:
    """Complete 3D trajectory of the ball."""
    points: List[TrajectoryPoint3D] = field(default_factory=list)
    bounces: List[BounceEvent] = field(default_factory=list)

    def get_positions(self) -> np.ndarray:
        """Get (N, 3) array of [x, y, z] positions."""
        if not self.points:
            return np.empty((0, 3))
        return np.array([[p.x, p.y, p.z] for p in self.points])

    def get_velocities(self) -> np.ndarray:
        """Get (N, 3) array of [vx, vy, vz] velocities."""
        if not self.points:
            return np.empty((0, 3))
        return np.array([[p.vx, p.vy, p.vz] for p in self.points])


# ============================================================================
# 2D -> Court Projection
# ============================================================================

class CourtProjector:
    """
    Projects 2D pixel coordinates to court coordinates using homography.

    The homography H maps image pixel (u, v) to court surface (x, y).
    This assumes the ball is on or near the court surface (z ~ 0).

    For balls in the air (z > 0), there will be projection error
    proportional to the ball's height.
    """

    def __init__(self, homography: np.ndarray):
        """
        Args:
            homography: 3x3 homography matrix (image -> court)
        """
        self.H = homography.astype(np.float64)
        self.H_inv = np.linalg.inv(self.H)

    def image_to_court(
        self, pixel_x: float, pixel_y: float
    ) -> Tuple[float, float]:
        """
        Project a 2D pixel coordinate to court surface coordinate.

        Args:
            pixel_x, pixel_y: Image pixel coordinates

        Returns:
            (court_x, court_y) in meters
        """
        return transform_point(self.H, np.array([pixel_x, pixel_y]))

    def court_to_image(
        self, court_x: float, court_y: float
    ) -> Tuple[float, float]:
        """
        Project a court coordinate back to image pixel.

        Args:
            court_x, court_y: Court coordinates in meters

        Returns:
            (pixel_x, pixel_y)
        """
        return transform_point(self.H_inv, np.array([court_x, court_y]))

    def project_trajectory(
        self, pixel_positions: np.ndarray
    ) -> np.ndarray:
        """
        Project multiple pixel positions to court coordinates.

        Args:
            pixel_positions: (N, 2) array of (pixel_x, pixel_y)

        Returns:
            (N, 2) array of (court_x, court_y)
        """
        return transform_points(self.H, pixel_positions)


# ============================================================================
# Physics-Based Height Estimation
# ============================================================================

class PhysicsModel:
    """
    Physics-based model for ball trajectory.

    Models the ball as a projectile under gravity:
        x(t) = x0 + vx0 * t
        y(t) = y0 + vy0 * t
        z(t) = z0 + vz0 * t - 0.5 * g * t^2

    The z coordinate (height) cannot be directly observed from
    a single camera, so it is inferred using:
    1. Physics constraints (gravity, bounce characteristics)
    2. Apparent ball size changes (closer to camera = larger)
    3. Trajectory curvature in image space
    """

    def __init__(self, fps: float = 30.0, gravity: float = GRAVITY):
        self.fps = fps
        self.dt = 1.0 / fps
        self.gravity = gravity
        self.coefficient_of_restitution = 0.75  # Tennis ball on hard court

    def estimate_3d_trajectory(
        self,
        court_positions: np.ndarray,
        frame_ids: np.ndarray,
    ) -> List[TrajectoryPoint3D]:
        """
        Estimate 3D trajectory from 2D court projections.

        Uses parabolic curve fitting between bounce points to
        estimate the height z at each frame.

        Args:
            court_positions: (N, 2) array of (court_x, court_y)
            frame_ids: (N,) array of frame indices

        Returns:
            List of TrajectoryPoint3D with estimated z values
        """
        n = len(court_positions)
        if n < 3:
            # Not enough points for trajectory estimation
            return [
                TrajectoryPoint3D(
                    frame_id=int(frame_ids[i]),
                    x=float(court_positions[i, 0]),
                    y=float(court_positions[i, 1]),
                    z=0.0,
                    is_estimated=True,
                )
                for i in range(n)
            ]

        # Estimate velocities using finite differences
        velocities = np.zeros_like(court_positions)
        for i in range(1, n):
            dt = (frame_ids[i] - frame_ids[i - 1]) * self.dt
            if dt > 0:
                velocities[i] = (court_positions[i] - court_positions[i - 1]) / dt

        # Estimate height using parabolic segments
        # Between bounces, z follows a parabola: z = z0 + vz0*t - 0.5*g*t^2
        heights = self._estimate_heights(court_positions, frame_ids)

        # Compute z-velocities
        vz = np.zeros(n)
        for i in range(1, n):
            dt = (frame_ids[i] - frame_ids[i - 1]) * self.dt
            if dt > 0:
                vz[i] = (heights[i] - heights[i - 1]) / dt

        points = []
        for i in range(n):
            dt_from_start = (frame_ids[i] - frame_ids[0]) * self.dt
            p = TrajectoryPoint3D(
                frame_id=int(frame_ids[i]),
                x=float(court_positions[i, 0]),
                y=float(court_positions[i, 1]),
                z=float(heights[i]),
                vx=float(velocities[i, 0]),
                vy=float(velocities[i, 1]),
                vz=float(vz[i]),
                is_estimated=True,
            )
            points.append(p)

        return points

    def _estimate_heights(
        self,
        positions: np.ndarray,
        frame_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate ball height at each frame using parabolic model.

        Fits z(t) = a*t^2 + b*t + c segments where a ~ -g/2.

        Uses the velocity profile of the court projection:
        when the ball is higher, it moves faster in image space
        (due to perspective projection from a camera above).
        """
        n = len(positions)
        heights = np.zeros(n)

        # Compute speed profile (2D court speed)
        speeds = np.zeros(n)
        for i in range(1, n):
            dt = (frame_ids[i] - frame_ids[i - 1]) * self.dt
            if dt > 0:
                displacement = np.linalg.norm(positions[i] - positions[i - 1])
                speeds[i] = displacement / dt

        # Find potential bounce points (local minima in speed)
        bounce_indices = self._find_bounce_candidates(speeds, frame_ids)

        if len(bounce_indices) < 2:
            # No clear bounces: assume a single parabolic arc
            t = (frame_ids - frame_ids[0]).astype(float) * self.dt
            # Fit parabola: z = -g/2 * (t - t_peak)^2 + z_peak
            t_mid = t[-1] / 2.0
            z_peak = 2.0  # Default peak height estimate (meters)
            for i in range(n):
                heights[i] = max(
                    0.0,
                    z_peak - 0.5 * self.gravity * (t[i] - t_mid) ** 2,
                )
        else:
            # Multiple bounces: fit parabola between consecutive bounces
            for seg_start, seg_end in zip(
                bounce_indices[:-1], bounce_indices[1:]
            ):
                t = np.arange(seg_end - seg_start + 1, dtype=float) * self.dt
                t_mid = t[-1] / 2.0

                # Estimate peak height from time between bounces
                flight_time = t[-1]
                z_peak = 0.5 * self.gravity * (flight_time / 2.0) ** 2

                for j, idx in enumerate(range(seg_start, seg_end + 1)):
                    heights[idx] = max(
                        0.0,
                        z_peak - 0.5 * self.gravity * (t[j] - t_mid) ** 2,
                    )

        return heights

    def _find_bounce_candidates(
        self,
        speeds: np.ndarray,
        frame_ids: np.ndarray,
        min_speed_ratio: float = 0.5,
    ) -> List[int]:
        """
        Find potential bounce points from speed profile.

        A bounce is characterized by:
        - Sudden speed reduction (ball decelerates on contact)
        - Direction change in vertical component
        """
        if len(speeds) < 5:
            return [0, len(speeds) - 1]

        bounces = [0]  # Start as virtual bounce

        # Smooth speeds
        kernel = np.ones(3) / 3
        smoothed = np.convolve(speeds, kernel, mode="same")

        # Find local minima in speed
        for i in range(2, len(smoothed) - 2):
            if (
                smoothed[i] < smoothed[i - 1]
                and smoothed[i] < smoothed[i + 1]
                and smoothed[i] < np.mean(smoothed) * min_speed_ratio
            ):
                bounces.append(i)

        bounces.append(len(speeds) - 1)  # End as virtual bounce
        return bounces


# ============================================================================
# Extended Kalman Filter for 3D Tracking
# ============================================================================

class ExtendedKalmanFilter3D:
    """
    Extended Kalman Filter for 3D ball trajectory estimation.

    State: x = [x, y, z, vx, vy, vz]^T (6D)

    Nonlinear dynamics (gravity):
        x_{k+1} = x_k + vx_k * dt
        y_{k+1} = y_k + vy_k * dt
        z_{k+1} = z_k + vz_k * dt - 0.5 * g * dt^2
        vx_{k+1} = vx_k
        vy_{k+1} = vy_k
        vz_{k+1} = vz_k - g * dt

    Observation: z_obs = [x, y]^T (2D court projection)

    The EKF linearizes the observation model at each step
    to handle the nonlinear relationship between 3D state
    and 2D observations.
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        gravity: float = GRAVITY,
        process_noise_pos: float = 0.1,
        process_noise_vel: float = 1.0,
        measurement_noise: float = 0.5,
    ):
        self.dt = dt
        self.gravity = gravity

        # State dimension
        self.dim_x = 6
        self.dim_z = 2  # observe (x, y) on court

        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros((self.dim_x, 1))

        # State covariance
        self.P = np.eye(self.dim_x) * 10.0

        # Process noise
        q_pos = process_noise_pos ** 2
        q_vel = process_noise_vel ** 2
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

        # Measurement noise
        r = measurement_noise ** 2
        self.R = np.diag([r, r])

        # Observation matrix (observe x, y only)
        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1.0  # observe x
        self.H[1, 1] = 1.0  # observe y

    def initialize(
        self, x: float, y: float, z: float = 1.0,
        vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
    ):
        """Set initial state."""
        self.x = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.P = np.eye(self.dim_x) * 10.0

    def predict(self) -> np.ndarray:
        """
        EKF Predict step with nonlinear dynamics (gravity).

        Returns predicted state.
        """
        dt = self.dt
        g = self.gravity

        # Nonlinear state transition
        x_new = np.zeros_like(self.x)
        x_new[0] = self.x[0] + self.x[3] * dt  # x += vx * dt
        x_new[1] = self.x[1] + self.x[4] * dt  # y += vy * dt
        x_new[2] = self.x[2] + self.x[5] * dt - 0.5 * g * dt ** 2  # z with gravity
        x_new[3] = self.x[3]  # vx constant
        x_new[4] = self.x[4]  # vy constant
        x_new[5] = self.x[5] - g * dt  # vz decreases due to gravity

        # Ensure z >= 0 (ball can't go below court)
        if x_new[2, 0] < 0:
            x_new[2, 0] = 0.0
            x_new[5, 0] = abs(x_new[5, 0]) * 0.75  # Bounce with energy loss

        self.x = x_new

        # Jacobian of state transition (F matrix)
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q

        return self.x.flatten()

    def update(self, court_x: float, court_y: float) -> np.ndarray:
        """
        EKF Update step with 2D court observation.

        Args:
            court_x, court_y: Observed court coordinates

        Returns:
            Updated state estimate.
        """
        z = np.array([[court_x], [court_y]])

        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

        return self.x.flatten()

    def get_state(self) -> Dict:
        """Get current state as a dictionary."""
        s = self.x.flatten()
        return {
            "x": s[0], "y": s[1], "z": s[2],
            "vx": s[3], "vy": s[4], "vz": s[5],
        }


# ============================================================================
# Bounce Detection
# ============================================================================

class BounceDetector:
    """
    Detects ball bounce events from the 3D trajectory.

    Methods:
    1. Height-based: z crosses zero (from positive to near-zero)
    2. Velocity-based: vz changes sign (downward -> upward)
    3. ML-based: Feature extraction + classifier (scikit-learn/CatBoost)

    Features for ML classifier:
    - Change in vertical direction (vz sign flip)
    - Speed change ratio before/after
    - Height at frame
    - Distance between consecutive positions
    - Acceleration magnitude
    """

    def __init__(
        self,
        z_threshold: float = 0.15,
        velocity_threshold: float = 0.5,
        min_frames_between_bounces: int = 5,
    ):
        self.z_threshold = z_threshold
        self.velocity_threshold = velocity_threshold
        self.min_frames_between = min_frames_between_bounces
        self.classifier = None

    def detect_bounces_physics(
        self, trajectory: Trajectory3D
    ) -> List[BounceEvent]:
        """
        Detect bounces using physics heuristics.

        A bounce occurs when:
        1. Height z is near zero (below threshold)
        2. Vertical velocity vz changes from negative to positive
        """
        bounces = []
        points = trajectory.points
        last_bounce_frame = -self.min_frames_between

        for i in range(1, len(points) - 1):
            prev_p = points[i - 1]
            curr_p = points[i]
            next_p = points[i + 1]

            # Check z near zero
            if curr_p.z > self.z_threshold:
                continue

            # Check vz sign change (downward -> upward)
            if prev_p.vz < 0 and next_p.vz > 0:
                # Ensure minimum gap between bounces
                if curr_p.frame_id - last_bounce_frame >= self.min_frames_between:
                    bounce = BounceEvent(
                        frame_id=curr_p.frame_id,
                        x=curr_p.x,
                        y=curr_p.y,
                        is_in=False,  # Will be classified later
                        confidence=0.8,
                    )
                    bounces.append(bounce)
                    curr_p.is_bounce = True
                    last_bounce_frame = curr_p.frame_id

        return bounces

    def detect_bounces_2d(
        self,
        court_positions: np.ndarray,
        frame_ids: np.ndarray,
    ) -> List[BounceEvent]:
        """
        Detect bounces from 2D court positions using trajectory features.

        Uses speed changes and direction changes as indicators.
        This is useful when 3D height estimation is unreliable.
        """
        n = len(court_positions)
        if n < 5:
            return []

        bounces = []
        last_bounce_idx = -self.min_frames_between

        # Compute velocities and accelerations
        velocities = np.diff(court_positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)

        # Compute direction changes
        for i in range(2, n - 2):
            if i - last_bounce_idx < self.min_frames_between:
                continue

            # Speed dip: speed decreases then increases
            if (
                speeds[i - 1] > speeds[i] * 1.3
                and speeds[i + 1] > speeds[i] * 1.3
            ):
                # Direction change
                v_before = velocities[i - 1]
                v_after = velocities[i]
                cos_angle = np.dot(v_before, v_after) / (
                    np.linalg.norm(v_before) * np.linalg.norm(v_after) + 1e-8
                )

                if cos_angle < 0.8:  # Significant direction change
                    bounce = BounceEvent(
                        frame_id=int(frame_ids[i]),
                        x=float(court_positions[i, 0]),
                        y=float(court_positions[i, 1]),
                        is_in=False,
                        confidence=float(1.0 - cos_angle),
                    )
                    bounces.append(bounce)
                    last_bounce_idx = i

        return bounces

    def extract_features(
        self,
        trajectory: Trajectory3D,
        window: int = 3,
    ) -> np.ndarray:
        """
        Extract features for ML-based bounce classification.

        For each frame, computes a feature vector from the surrounding
        trajectory window. Used for training a classifier.

        Features per frame:
        - z value
        - vz value
        - vz sign change indicator
        - speed before / after ratio
        - direction change angle
        - acceleration magnitude
        - distance to previous point
        """
        points = trajectory.points
        n = len(points)
        features = []

        for i in range(window, n - window):
            f = []

            # Height and velocity
            f.append(points[i].z)
            f.append(points[i].vz)

            # vz sign change
            vz_before = points[i - 1].vz
            vz_after = points[i + 1].vz if i + 1 < n else points[i].vz
            f.append(1.0 if vz_before * vz_after < 0 else 0.0)

            # Speed before and after
            speed_before = np.sqrt(
                points[i - 1].vx ** 2 + points[i - 1].vy ** 2 + points[i - 1].vz ** 2
            )
            speed_after = np.sqrt(
                points[i].vx ** 2 + points[i].vy ** 2 + points[i].vz ** 2
            )
            speed_ratio = speed_after / (speed_before + 1e-8)
            f.append(speed_ratio)

            # Direction change
            v_before = np.array([points[i - 1].vx, points[i - 1].vy])
            v_after = np.array([points[i].vx, points[i].vy])
            norm_b = np.linalg.norm(v_before) + 1e-8
            norm_a = np.linalg.norm(v_after) + 1e-8
            cos_angle = np.dot(v_before, v_after) / (norm_b * norm_a)
            f.append(float(cos_angle))

            # Acceleration
            ax = points[i].vx - points[i - 1].vx
            ay = points[i].vy - points[i - 1].vy
            az = points[i].vz - points[i - 1].vz
            f.append(np.sqrt(ax ** 2 + ay ** 2 + az ** 2))

            # Distance to previous point
            dx = points[i].x - points[i - 1].x
            dy = points[i].y - points[i - 1].y
            dz = points[i].z - points[i - 1].z
            f.append(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))

            features.append(f)

        return np.array(features) if features else np.empty((0, 7))


# ============================================================================
# In/Out Classification
# ============================================================================

class InOutClassifier:
    """
    Classifies whether a bounce point is inside or outside court boundaries.

    Simple geometric check: compare bounce (x, y) coordinates against
    the standard court dimensions.

    Also provides margin-of-error estimation based on the uncertainty
    from the homography and trajectory estimation.
    """

    def __init__(
        self,
        court_type: str = "tennis_singles",
        margin: float = 0.0,
    ):
        """
        Args:
            court_type: "tennis_singles", "tennis_doubles", or "pickleball"
            margin: Additional margin in meters (for ball radius, etc.)
        """
        if court_type == "tennis_singles":
            self.bounds = TENNIS_SINGLES_BOUNDS.copy()
        elif court_type == "tennis_doubles":
            self.bounds = TENNIS_DOUBLES_BOUNDS.copy()
        elif court_type == "pickleball":
            self.bounds = {
                "x_min": 0.0,
                "x_max": 13.41,
                "y_min": 0.0,
                "y_max": 6.10,
            }
        else:
            self.bounds = TENNIS_DOUBLES_BOUNDS.copy()

        # Apply ball radius margin
        self.margin = margin + TENNIS_BALL_RADIUS

    def classify(self, bounce: BounceEvent) -> bool:
        """
        Determine if a bounce is in or out.

        The ball is "in" if the bounce point (including ball radius)
        is within the court boundaries.

        Args:
            bounce: BounceEvent with (x, y) coordinates

        Returns:
            True if ball is in, False if out
        """
        x, y = bounce.x, bounce.y
        is_in = (
            x >= self.bounds["x_min"] - self.margin
            and x <= self.bounds["x_max"] + self.margin
            and y >= self.bounds["y_min"] - self.margin
            and y <= self.bounds["y_max"] + self.margin
        )
        return is_in

    def classify_with_confidence(
        self, bounce: BounceEvent, position_uncertainty: float = 0.1
    ) -> Tuple[bool, float]:
        """
        Classify with confidence based on distance to boundary.

        Returns:
            (is_in, confidence) where confidence is higher when
            the ball is clearly in or clearly out.
        """
        x, y = bounce.x, bounce.y

        # Distance to nearest boundary
        dx_min = x - (self.bounds["x_min"] - self.margin)
        dx_max = (self.bounds["x_max"] + self.margin) - x
        dy_min = y - (self.bounds["y_min"] - self.margin)
        dy_max = (self.bounds["y_max"] + self.margin) - y

        min_distance = min(dx_min, dx_max, dy_min, dy_max)
        is_in = min_distance >= 0

        # Confidence based on distance to boundary relative to uncertainty
        confidence = min(1.0, abs(min_distance) / (position_uncertainty + 1e-8))
        confidence = max(0.5, confidence)  # Floor at 50%

        return is_in, confidence

    def classify_bounces(
        self, bounces: List[BounceEvent]
    ) -> List[BounceEvent]:
        """Classify all bounce events."""
        for bounce in bounces:
            is_in, conf = self.classify_with_confidence(bounce)
            bounce.is_in = is_in
            bounce.confidence = conf
        return bounces


# ============================================================================
# Full 3D Reconstruction Pipeline
# ============================================================================

class TrajectoryReconstructor:
    """
    Full 3D trajectory reconstruction pipeline.

    Combines:
    - Court projection (homography)
    - Physics-based height estimation
    - Extended Kalman Filter
    - Bounce detection
    - In/Out classification
    """

    def __init__(
        self,
        homography: np.ndarray,
        fps: float = 30.0,
        court_type: str = "tennis_singles",
    ):
        self.projector = CourtProjector(homography)
        self.physics = PhysicsModel(fps=fps)
        self.ekf = ExtendedKalmanFilter3D(dt=1.0 / fps)
        self.bounce_detector = BounceDetector()
        self.in_out = InOutClassifier(court_type=court_type)
        self.fps = fps

        self.trajectory = Trajectory3D()
        self.initialized = False

    def process_frame(
        self,
        frame_id: int,
        ball_pixel_x: Optional[float],
        ball_pixel_y: Optional[float],
    ) -> Optional[TrajectoryPoint3D]:
        """
        Process a single frame: project to court, estimate 3D, update EKF.

        Args:
            frame_id: Current frame number
            ball_pixel_x, ball_pixel_y: Ball position in pixels (None if not detected)

        Returns:
            TrajectoryPoint3D or None
        """
        if ball_pixel_x is None or ball_pixel_y is None:
            # No detection: EKF predict only
            if self.initialized:
                state = self.ekf.predict()
                point = TrajectoryPoint3D(
                    frame_id=frame_id,
                    x=state[0], y=state[1], z=max(0, state[2]),
                    vx=state[3], vy=state[4], vz=state[5],
                    is_estimated=True,
                )
                self.trajectory.points.append(point)
                return point
            return None

        # Project to court coordinates
        court_x, court_y = self.projector.image_to_court(
            ball_pixel_x, ball_pixel_y
        )

        if not self.initialized:
            # Initialize EKF with first observation
            self.ekf.initialize(court_x, court_y, z=1.0)
            self.initialized = True

        # EKF predict + update
        self.ekf.predict()
        state = self.ekf.update(court_x, court_y)

        point = TrajectoryPoint3D(
            frame_id=frame_id,
            x=state[0], y=state[1], z=max(0, state[2]),
            vx=state[3], vy=state[4], vz=state[5],
        )
        self.trajectory.points.append(point)

        return point

    def finalize(self) -> Trajectory3D:
        """
        Finalize trajectory: detect bounces and classify in/out.

        Call this after processing all frames.
        """
        # Detect bounces
        bounces = self.bounce_detector.detect_bounces_physics(self.trajectory)

        # If no physics bounces found, try 2D method
        if not bounces and len(self.trajectory.points) > 5:
            positions = np.array([
                [p.x, p.y] for p in self.trajectory.points
            ])
            frame_ids = np.array([
                p.frame_id for p in self.trajectory.points
            ])
            bounces = self.bounce_detector.detect_bounces_2d(
                positions, frame_ids
            )

        # Classify in/out
        bounces = self.in_out.classify_bounces(bounces)
        self.trajectory.bounces = bounces

        return self.trajectory

    def reset(self):
        """Reset the reconstructor for a new rally."""
        self.trajectory = Trajectory3D()
        self.ekf = ExtendedKalmanFilter3D(dt=1.0 / self.fps)
        self.initialized = False
