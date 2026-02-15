"""
In/Out Classification Module
=============================

Provides high-level in/out classification combining trajectory analysis,
bounce detection, and ML-based classification.

This module wraps the functionality from trajectory_3d.py and adds:
- ML-based bounce classification using scikit-learn / CatBoost
- Confidence calibration
- Batch processing for multiple rallies
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import os

try:
    from .trajectory_3d import (
        BounceEvent,
        BounceDetector,
        InOutClassifier,
        Trajectory3D,
        TrajectoryPoint3D,
    )
except ImportError:
    from trajectory_3d import (
        BounceEvent,
        BounceDetector,
        InOutClassifier,
        Trajectory3D,
        TrajectoryPoint3D,
    )


class MLBounceClassifier:
    """
    Machine Learning-based bounce detector.

    Trains a binary classifier on trajectory features to predict
    whether each frame is a bounce frame or not.

    Supports:
    - scikit-learn classifiers (RandomForest, GradientBoosting)
    - CatBoost (if installed)
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Args:
            model_type: "random_forest", "gradient_boosting", or "catboost"
        """
        self.model_type = model_type
        self.model = None
        self._build_model()

    def _build_model(self):
        """Initialize the ML model."""
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight="balanced",
            )
        elif self.model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        elif self.model_type == "catboost":
            try:
                from catboost import CatBoostClassifier
                self.model = CatBoostClassifier(
                    iterations=200,
                    depth=6,
                    learning_rate=0.1,
                    verbose=0,
                    auto_class_weights="Balanced",
                )
            except ImportError:
                print("CatBoost not installed, falling back to RandomForest")
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=100, random_state=42
                )

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Train the bounce classifier.

        Args:
            features: (N, 7) feature matrix from BounceDetector.extract_features
            labels: (N,) binary labels (1=bounce, 0=not bounce)
        """
        self.model.fit(features, labels)

    def predict(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict bounce probability for each frame.

        Returns:
            predictions: (N,) binary predictions
            probabilities: (N,) bounce probabilities
        """
        predictions = self.model.predict(features)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[:, 1]
        else:
            probabilities = predictions.astype(float)
        return predictions, probabilities

    def save(self, path: str):
        """Save model to file."""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load model from file."""
        import joblib
        self.model = joblib.load(path)


class EnhancedInOutSystem:
    """
    Complete In/Out decision system combining physics and ML approaches.

    Pipeline:
    1. Physics-based bounce detection (heuristic)
    2. ML-based bounce detection (learned)
    3. Ensemble: combine both for higher accuracy
    4. In/Out classification for each detected bounce
    5. Confidence estimation
    """

    def __init__(
        self,
        court_type: str = "tennis_singles",
        ml_model_path: Optional[str] = None,
    ):
        self.bounce_detector = BounceDetector()
        self.in_out_classifier = InOutClassifier(court_type=court_type)
        self.ml_classifier: Optional[MLBounceClassifier] = None

        if ml_model_path and os.path.exists(ml_model_path):
            self.ml_classifier = MLBounceClassifier()
            self.ml_classifier.load(ml_model_path)

    def analyze_trajectory(
        self, trajectory: Trajectory3D
    ) -> Dict:
        """
        Full analysis of a ball trajectory.

        Returns dict with:
        - bounces: list of BounceEvent
        - in_out_decisions: list of (BounceEvent, is_in, confidence)
        - summary: overall statistics
        """
        # Physics-based bounce detection
        physics_bounces = self.bounce_detector.detect_bounces_physics(trajectory)

        # ML-based bounce detection (if model available)
        ml_bounces = []
        if self.ml_classifier is not None:
            features = self.bounce_detector.extract_features(trajectory)
            if len(features) > 0:
                preds, probs = self.ml_classifier.predict(features)
                window = 3  # same as used in extract_features
                for i, (pred, prob) in enumerate(zip(preds, probs)):
                    if pred == 1:
                        idx = i + window
                        if idx < len(trajectory.points):
                            p = trajectory.points[idx]
                            ml_bounces.append(BounceEvent(
                                frame_id=p.frame_id,
                                x=p.x, y=p.y,
                                is_in=False,
                                confidence=float(prob),
                            ))

        # Ensemble: merge physics and ML bounces
        all_bounces = self._merge_bounces(physics_bounces, ml_bounces)

        # Classify in/out
        decisions = []
        for bounce in all_bounces:
            is_in, confidence = self.in_out_classifier.classify_with_confidence(bounce)
            bounce.is_in = is_in
            bounce.confidence = confidence
            decisions.append({
                "bounce": bounce,
                "is_in": is_in,
                "confidence": confidence,
            })

        # Summary statistics
        total_bounces = len(all_bounces)
        in_count = sum(1 for b in all_bounces if b.is_in)
        out_count = total_bounces - in_count

        return {
            "bounces": all_bounces,
            "decisions": decisions,
            "summary": {
                "total_bounces": total_bounces,
                "in_count": in_count,
                "out_count": out_count,
                "trajectory_length": len(trajectory.points),
            },
        }

    def _merge_bounces(
        self,
        physics_bounces: List[BounceEvent],
        ml_bounces: List[BounceEvent],
        frame_tolerance: int = 3,
    ) -> List[BounceEvent]:
        """
        Merge physics and ML bounce detections.

        If both methods detect a bounce within frame_tolerance frames,
        keep the one with higher confidence (boost confidence).
        """
        if not ml_bounces:
            return physics_bounces
        if not physics_bounces:
            return ml_bounces

        merged = []
        used_ml = set()

        for pb in physics_bounces:
            best_match = None
            best_dist = float("inf")

            for j, mb in enumerate(ml_bounces):
                dist = abs(pb.frame_id - mb.frame_id)
                if dist <= frame_tolerance and dist < best_dist:
                    best_dist = dist
                    best_match = j

            if best_match is not None:
                # Both methods agree: boost confidence
                mb = ml_bounces[best_match]
                merged_bounce = BounceEvent(
                    frame_id=pb.frame_id,
                    x=(pb.x + mb.x) / 2,
                    y=(pb.y + mb.y) / 2,
                    is_in=False,
                    confidence=min(1.0, (pb.confidence + mb.confidence) / 2 + 0.1),
                )
                merged.append(merged_bounce)
                used_ml.add(best_match)
            else:
                merged.append(pb)

        # Add unmatched ML bounces with lower confidence
        for j, mb in enumerate(ml_bounces):
            if j not in used_ml:
                mb.confidence *= 0.7  # Lower confidence for ML-only
                merged.append(mb)

        # Sort by frame_id
        merged.sort(key=lambda b: b.frame_id)
        return merged
