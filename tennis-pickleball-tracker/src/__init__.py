"""
Tennis/Pickleball Detection & Tracking System
=============================================

A computer vision pipeline for detecting and tracking balls and players
in tennis/pickleball videos using a single camera.

Modules:
    - court_detection: Court line detection and homography estimation
    - object_detection: Ball and player detection (YOLO, TrackNet, HOG)
    - object_tracking: Kalman Filter, Optical Flow, DeepSORT tracking
    - trajectory_3d: 3D trajectory reconstruction, bounce & in/out
    - visualization: Overlay trajectories, mini-map, heatmaps
    - pipeline: End-to-end integrated pipeline
"""

__version__ = "1.0.0"
