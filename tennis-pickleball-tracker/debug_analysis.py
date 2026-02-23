"""Debug script to analyze court polygon and player scoring."""
import sys, os, cv2, numpy as np
sys.path.insert(0, "src")
os.chdir(r"D:\Downloads\cv\tennis-pickleball-tracker")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from court_detection import ClassicalCourtDetector
from object_detection import YOLODetector

cap = cv2.VideoCapture("data/raw/Video Project 4.mp4")
ret, f0 = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
ret, f1 = cap.read()
cap.release()
h, w = f1.shape[:2]
print(f"Frame size: {w}x{h}")

# Court detection on frame 1
cd = ClassicalCourtDetector()
H, res = cd.detect_and_compute_homography(f1)
print(f"Homography found: {H is not None}")
print(f"Horizontal lines: {len(res['horizontal'])}")
print(f"Vertical lines: {len(res['vertical'])}")
print(f"Intersections: {len(res['intersections'])}")
for i, pt in enumerate(res["intersections"]):
    print(f"  [{i}] ({pt[0]:.0f}, {pt[1]:.0f})")

# Convex hull of intersections
pts = np.array(res["intersections"], dtype=np.float32)
hull = cv2.convexHull(pts)
print(f"\nConvex hull ({len(hull)} points):")
for p in hull:
    print(f"  ({p[0][0]:.0f}, {p[0][1]:.0f})")

# Draw debug image: court lines + hull + player detections
debug = f1.copy()
# Draw intersections
for pt in res["intersections"]:
    cv2.circle(debug, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
# Draw hull
cv2.polylines(debug, [hull.astype(np.int32)], True, (0, 255, 0), 2)

# YOLO on frame 74
cap = cv2.VideoCapture("data/raw/Video Project 4.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 74)
ret, frame74 = cap.read()
cap.release()

d = YOLODetector(model_path="yolov8s.pt", device="cpu")
dets = d.detect(frame74)
players = [x for x in dets if x.class_name == "player"]
polygon = hull.reshape(-1, 2).astype(np.int32)

print(f"\nFrame 74 - {len(players)} players:")
for det in sorted(players, key=lambda d: d.confidence, reverse=True):
    x1, y1, x2, y2 = det.bbox
    foot_x = (x1 + x2) / 2.0
    foot_y = float(y2)
    area = int((x2 - x1) * (y2 - y1))
    dist = cv2.pointPolygonTest(polygon, (foot_x, foot_y), measureDist=True)
    cx = (x1 + x2) / 2.0
    center_score = 1.0 - abs(cx - w / 2.0) / (w / 2.0)
    size_score = min(1.0, area / 30000)
    score = det.confidence * 0.3 + size_score * 0.4 + center_score * 0.3
    on_court = "IN" if dist >= -80 else "OUT"
    print(
        f"  conf={det.confidence:.2f} bbox=({x1},{y1},{x2},{y2}) "
        f"foot=({foot_x:.0f},{foot_y:.0f}) area={area:6d} "
        f"dist={dist:6.0f} {on_court:3s} cscore={center_score:.2f} "
        f"total={score:.3f}"
    )

# Draw on frame74
debug74 = frame74.copy()
cv2.polylines(debug74, [polygon], True, (0, 255, 0), 2)
for det in players:
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(debug74, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)
    cv2.circle(debug74, (foot_x, foot_y), 4, (0, 0, 255), -1)

cv2.imwrite("outputs/test_results/debug_court_polygon.jpg", debug)
cv2.imwrite("outputs/test_results/debug_players_f74.jpg", debug74)
print("\nSaved debug images")
