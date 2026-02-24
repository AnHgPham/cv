"""
Generate debug images showing homography failure cases.
Produces clear visual evidence for professor review.
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r"D:\Downloads\cv\tennis-pickleball-tracker\src")
os.chdir(r"D:\Downloads\cv\tennis-pickleball-tracker")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from court_detection import ClassicalCourtDetector, TENNIS_COURT_CORNERS

OUT = "outputs/debug_for_professor"
os.makedirs(OUT, exist_ok=True)

# Read frames
cap = cv2.VideoCapture("data/raw/Video Project 4.mp4")
frames = {}
for i in range(300):
    ret, f = cap.read()
    if not ret:
        break
    if i in [0, 30, 60, 74, 100, 150, 200, 250]:
        frames[i] = f.copy()
cap.release()

court = ClassicalCourtDetector()

# Find best frame where H is computed
print("Scanning frames for homography status...")
h_frame = None
h_frame_num = None
for fnum in sorted(frames.keys()):
    f = frames[fnum]
    H_test, det_test = court.detect_and_compute_homography(f)
    r = court.detect(f)
    status = "H OK" if H_test is not None else "H=None"
    corners_str = ""
    if det_test.get("selected_corners") is not None:
        c = det_test["selected_corners"]
        corners_str = f" corners=({int(c[0][0])},{int(c[0][1])}),({int(c[1][0])},{int(c[1][1])}),({int(c[2][0])},{int(c[2][1])}),({int(c[3][0])},{int(c[3][1])})"
    print(f"  Frame {fnum}: {len(r['horizontal'])}H+{len(r['vertical'])}V, "
          f"{len(r['intersections'])} pts, {status}{corners_str}")
    if H_test is not None and h_frame is None:
        h_frame = f
        h_frame_num = fnum
        H_found = H_test
        det_found = det_test
        r_found = r

frame74 = frames[74]
fh, fw = frame74.shape[:2]

# =========================================================================
# IMAGE 1: Frame 74 - All detected lines & clustered intersections  
# =========================================================================
print(f"\n=== Image 1: Detected Lines (Frame 74) ===")
result = court.detect(frame74)
intersections = result["intersections"]
h_lines = result["horizontal"]
v_lines = result["vertical"]

img1 = frame74.copy()
overlay = img1.copy()
cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
img1 = cv2.addWeighted(img1, 0.7, overlay, 0.3, 0)

# Draw horizontal lines with extensions
for line in h_lines:
    x1, y1, x2, y2 = line
    cv2.line(img1, (x1, y1), (x2, y2), (255, 100, 0), 4)
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) > 1:
        ratio = 3000 / abs(dx)
        cv2.line(img1, (int(x1 - dx*ratio), int(y1 - dy*ratio)),
                 (int(x2 + dx*ratio), int(y2 + dy*ratio)), (255, 100, 0), 1)

# Draw vertical lines with extensions
for line in v_lines:
    x1, y1, x2, y2 = line
    cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 100), 4)
    dx, dy = x2 - x1, y2 - y1
    if abs(dy) > 1:
        ratio = 3000 / abs(dy)
        cv2.line(img1, (int(x1 - dx*ratio), int(y1 - dy*ratio)),
                 (int(x2 + dx*ratio), int(y2 + dy*ratio)), (0, 255, 100), 1)

# Draw intersections
for i, pt in enumerate(intersections):
    x, y = int(pt[0]), int(pt[1])
    clx = max(0, min(fw-1, x))
    cly = max(0, min(fh-1, y))
    cv2.circle(img1, (clx, cly), 12, (0, 255, 255), -1)
    cv2.circle(img1, (clx, cly), 13, (0, 0, 0), 2)
    # Only label a subset to avoid clutter
    if i in [0, 3, 6, 9, 12]:
        cv2.putText(img1, f"({int(pt[0])},{int(pt[1])})", (clx+16, cly+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Stats on intersection spread
xs = [pt[0] for pt in intersections]
ys = [pt[1] for pt in intersections]
x_spread = max(xs) - min(xs) if xs else 0

# Legend
cv2.rectangle(img1, (10, 10), (600, 165), (0, 0, 0), -1)
cv2.putText(img1, f"Frame 74: {len(h_lines)} horizontal (blue) + {len(v_lines)} vertical (green)",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(img1, f"{len(intersections)} intersections (yellow dots)",
            (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
cv2.putText(img1, f"X range: {int(min(xs))}-{int(max(xs))} (spread={int(x_spread)}px)",
            (20, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(img1, f"Y range: {int(min(ys))}-{int(max(ys))} (spread={int(max(ys)-min(ys))}px)",
            (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
cv2.putText(img1, "PROBLEM: All intersections in narrow x-band (~55px)",
            (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.putText(img1, "Cannot form 4-corner quadrilateral -> H = None",
            (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite(f"{OUT}/1_frame74_intersections.jpg", img1)
print(f"  Saved: {OUT}/1_frame74_intersections.jpg")

# =========================================================================
# IMAGE 2: Frame with H computed - corners selected vs expected
# =========================================================================
if h_frame is not None:
    print(f"\n=== Image 2: Corner Selection (Frame {h_frame_num}) ===")
    img2 = h_frame.copy()
    overlay = img2.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    img2 = cv2.addWeighted(img2, 0.7, overlay, 0.3, 0)

    # Draw all intersections
    for pt in r_found["intersections"]:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < fw and 0 <= y < fh:
            cv2.circle(img2, (x, y), 7, (120, 120, 120), -1)
            cv2.putText(img2, f"({x},{y})", (x+10, y+3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    # Draw selected corners
    corners = det_found["selected_corners"]
    labels = ["TL", "TR", "BL", "BR"]
    colors = [(255, 50, 50), (50, 255, 50), (50, 50, 255), (0, 255, 255)]
    for i, (corner, label, color) in enumerate(zip(corners, labels, colors)):
        x, y = int(corner[0]), int(corner[1])
        cv2.drawMarker(img2, (x, y), color, cv2.MARKER_CROSS, 30, 3)
        cv2.putText(img2, f"Selected {label} ({x},{y})", (x+20, y-5 if i < 2 else y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Draw the quadrilateral they form
    quad = np.array([corners[0], corners[1], corners[3], corners[2]], dtype=np.int32)
    cv2.polylines(img2, [quad.reshape(-1, 1, 2)], True, (0, 165, 255), 2)

    # Legend
    cv2.rectangle(img2, (10, 10), (650, 100), (0, 0, 0), -1)
    cv2.putText(img2, f"Frame {h_frame_num}: 4 selected corners for homography",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(img2, "Orange quad = detected court region (from these 4 corners)",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.putText(img2, "These corners may NOT correspond to actual court corners!",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(f"{OUT}/2_corner_selection.jpg", img2)
    print(f"  Saved: {OUT}/2_corner_selection.jpg")

    # =========================================================================
    # IMAGE 3: Homography reprojection
    # =========================================================================
    print(f"\n=== Image 3: Homography Reprojection (Frame {h_frame_num}) ===")
    img3 = h_frame.copy()
    overlay = img3.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    img3 = cv2.addWeighted(img3, 0.7, overlay, 0.3, 0)

    H_inv = np.linalg.inv(H_found)
    court_pts = np.array([
        [0.0, 0.0], [0.0, 10.97], [23.77, 10.97], [23.77, 0.0]
    ], dtype=np.float32)
    proj_pts = cv2.perspectiveTransform(
        court_pts.reshape(-1, 1, 2), H_inv
    ).reshape(-1, 2)

    # Frame boundary
    cv2.rectangle(img3, (3, 3), (fw-4, fh-4), (255, 255, 0), 3)

    # Projected polygon
    draw_pts = proj_pts.copy()
    draw_pts[:, 0] = np.clip(draw_pts[:, 0], -1000, fw+1000)
    draw_pts[:, 1] = np.clip(draw_pts[:, 1], -1000, fh+1000)
    poly_order = np.array([draw_pts[0], draw_pts[1], draw_pts[2], draw_pts[3]], dtype=np.int32)
    cv2.polylines(img3, [poly_order.reshape(-1, 1, 2)], True, (0, 0, 255), 3)

    # Label each projected corner
    corner_labels = [
        ("Court (0,0)", (255, 50, 50)),
        ("Court (0,10.97)", (50, 255, 50)),
        ("Court (23.77,10.97)", (0, 255, 255)),
        ("Court (23.77,0)", (255, 50, 255)),
    ]
    for i, ((label, color), pt) in enumerate(zip(corner_labels, proj_pts)):
        x, y = int(pt[0]), int(pt[1])
        cx_draw = max(5, min(fw-10, x))
        cy_draw = max(20, min(fh-10, y))
        cv2.circle(img3, (cx_draw, cy_draw), 10, color, -1)
        offset_y = 25 if i % 2 == 0 else -10
        cv2.putText(img3, f"{label} -> px({x},{y})",
                    (max(5, min(fw-300, cx_draw+15)), max(20, min(fh-5, cy_draw+offset_y))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # Court center projection
    center_court = np.array([[[11.885, 5.485]]], dtype=np.float32)
    center_px = cv2.perspectiveTransform(center_court, H_inv).reshape(2)
    ccx, ccy = int(center_px[0]), int(center_px[1])
    if 0 <= ccx < fw and 0 <= ccy < fh:
        cv2.drawMarker(img3, (ccx, ccy), (255, 255, 255), cv2.MARKER_STAR, 20, 2)
        cv2.putText(img3, f"Center(11.9,5.5)->px({ccx},{ccy})", (ccx+15, ccy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    print(f"  H matrix:\n{H_found}")
    print(f"  Projected court corners:")
    for cpt, ppt in zip(court_pts, proj_pts):
        inside = 0 <= ppt[0] <= fw and 0 <= ppt[1] <= fh
        print(f"    ({cpt[0]:.1f},{cpt[1]:.1f}) -> pixel ({ppt[0]:.0f},{ppt[1]:.0f}) "
              f"{'[in frame]' if inside else '[OUT OF FRAME!]'}")

    # Legend
    cv2.rectangle(img3, (10, fh-100), (720, fh-10), (0, 0, 0), -1)
    cv2.putText(img3, f"Frame {h_frame_num}: Red = court (23.77x10.97m) projected back via H^-1",
                (20, fh-75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img3, "Yellow = frame bounds. Red polygon should match visible court lines",
                (20, fh-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(img3, "If polygon doesn't match -> homography is WRONG",
                (20, fh-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

    cv2.imwrite(f"{OUT}/3_homography_reprojection.jpg", img3)
    print(f"  Saved: {OUT}/3_homography_reprojection.jpg")

    # =========================================================================
    # IMAGE 4: Player foot projections
    # =========================================================================
    print(f"\n=== Image 4: Player Foot Projections (Frame {h_frame_num}) ===")
    img4 = h_frame.copy()
    overlay = img4.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    img4 = cv2.addWeighted(img4, 0.7, overlay, 0.3, 0)

    from object_detection import YOLODetector
    yolo = YOLODetector(model_path="yolov8s.pt", conf_threshold=0.25,
                        device="cpu", is_custom_model=False)
    dets = yolo.detect(h_frame)
    players = [d for d in dets if d.class_name in ("player", "person")]

    print(f"  Detected {len(players)} people")
    for det in players:
        x1, y1, x2, y2 = det.bbox
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)

        pt = np.array([[[foot_x, foot_y]]], dtype=np.float32)
        court_pt = cv2.perspectiveTransform(pt, H_found)
        cx, cy = float(court_pt[0, 0, 0]), float(court_pt[0, 0, 1])

        in_court = (0 <= cx <= 23.77 and 0 <= cy <= 10.97)
        color = (0, 255, 0) if in_court else (0, 0, 255)
        status = "IN COURT" if in_court else "OUTSIDE!"

        cv2.rectangle(img4, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img4, (int(foot_x), int(foot_y)), 8, color, -1)

        ty = max(25, y1 - 25)
        cv2.rectangle(img4, (x1-2, ty-15), (x1+300, ty+22), (0, 0, 0), -1)
        cv2.putText(img4, f"Foot:({int(foot_x)},{int(foot_y)}) -> Court:({cx:.1f},{cy:.1f})m",
                    (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(img4, status, (x1, ty+17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"    ({int(foot_x)},{int(foot_y)}) -> ({cx:.2f},{cy:.2f})m [{status}]")

    cv2.rectangle(img4, (10, fh-80), (680, fh-10), (0, 0, 0), -1)
    cv2.putText(img4, f"Frame {h_frame_num}: Player feet projected through H to court coordinates",
                (20, fh-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img4, "Court: 23.77m x 10.97m. Green=valid, Red=OUTSIDE (wrong H)",
                (20, fh-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

    cv2.imwrite(f"{OUT}/4_player_projection.jpg", img4)
    print(f"  Saved: {OUT}/4_player_projection.jpg")

# =========================================================================
# IMAGE 5: Multi-frame comparison grid
# =========================================================================
print("\n=== Image 5: Multi-frame Comparison ===")
grid_frames = [0, 30, 74, 150]
cell_w, cell_h = fw // 2, fh // 2
img5 = np.zeros((fh, fw, 3), dtype=np.uint8)

for idx, fnum in enumerate(grid_frames):
    if fnum not in frames:
        continue
    f = frames[fnum]
    r = court.detect(f)
    vis = f.copy()
    overlay2 = vis.copy()
    cv2.rectangle(overlay2, (0, 0), (fw, fh), (0, 0, 0), -1)
    vis = cv2.addWeighted(vis, 0.7, overlay2, 0.3, 0)

    for line in r["horizontal"]:
        x1, y1, x2, y2 = line
        cv2.line(vis, (x1, y1), (x2, y2), (255, 100, 0), 3)
    for line in r["vertical"]:
        x1, y1, x2, y2 = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 100), 3)
    for pt in r["intersections"]:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < fw and 0 <= y < fh:
            cv2.circle(vis, (x, y), 10, (0, 255, 255), -1)

    H2, det2 = court.detect_and_compute_homography(f)
    h_status = "H computed" if H2 is not None else "H = None (FAIL)"
    h_color = (0, 255, 0) if H2 is not None else (0, 0, 255)

    if H2 is not None and det2.get("selected_corners") is not None:
        for c in det2["selected_corners"]:
            cv2.drawMarker(vis, (int(c[0]), int(c[1])), (0, 0, 255), cv2.MARKER_CROSS, 20, 3)

    cv2.rectangle(vis, (5, 5), (500, 70), (0, 0, 0), -1)
    cv2.putText(vis, f"Frame {fnum}: {len(r['horizontal'])}H + {len(r['vertical'])}V lines, "
                f"{len(r['intersections'])} intersections",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(vis, h_status, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, h_color, 2)

    row, col = idx // 2, idx % 2
    small = cv2.resize(vis, (cell_w, cell_h))
    img5[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w] = small
    print(f"  Frame {fnum}: {len(r['horizontal'])}H+{len(r['vertical'])}V, "
          f"{len(r['intersections'])} pts, {h_status}")

# Center label
cv2.rectangle(img5, (cell_w-220, cell_h-20), (cell_w+220, cell_h+20), (0, 0, 0), -1)
cv2.putText(img5, "Line detection varies across frames -> unstable homography",
            (cell_w-215, cell_h+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

cv2.imwrite(f"{OUT}/5_multiframe_comparison.jpg", img5)
print(f"  Saved: {OUT}/5_multiframe_comparison.jpg")

# =========================================================================
# IMAGE 6: Side-by-side expected vs detected court region
# =========================================================================
print("\n=== Image 6: Expected vs Detected ===")
img6 = frame74.copy()

# Manually annotated court corners for frame 74 (accurate pixel coordinates)
# The main court's visible corners in perspective
expected = np.array([
    [835, 265],    # far-left baseline corner (top-left)
    [1185, 265],   # far-right baseline corner (top-right)
    [1635, 895],   # near-right baseline corner (bottom-right)
    [470, 880],    # near-left baseline corner (bottom-left)
], dtype=np.int32)

mask_green = np.zeros_like(img6)
cv2.fillPoly(mask_green, [expected], (0, 180, 0))
img6 = cv2.addWeighted(img6, 1.0, mask_green, 0.25, 0)
cv2.polylines(img6, [expected], True, (0, 255, 0), 3)

# Mark the intersection cluster zone (red rectangle showing where all detections are)
xs = [pt[0] for pt in intersections]
ys = [pt[1] for pt in intersections]
det_zone = (int(min(xs))-20, int(min(ys))-20, int(max(xs))+20, int(max(ys))+20)
cv2.rectangle(img6, (det_zone[0], det_zone[1]), (det_zone[2], det_zone[3]), (0, 0, 255), 3)

# Draw intersections
for pt in intersections:
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(img6, (x, y), 8, (0, 0, 255), -1)

# Labels
cv2.rectangle(img6, (10, 10), (700, 115), (0, 0, 0), -1)
cv2.putText(img6, "Green polygon = actual main court (what we NEED to detect)",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
cv2.putText(img6, "Red box = where ALL 15 intersections cluster (narrow ~55px strip)",
            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
cv2.putText(img6, "Problem: Detected lines only cover CENTER of court, miss baselines & sidelines",
            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
cv2.putText(img6, "Adjacent courts' lines add noise; main court boundaries not fully detected",
            (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

cv2.imwrite(f"{OUT}/6_expected_vs_detected.jpg", img6)
print(f"  Saved: {OUT}/6_expected_vs_detected.jpg")

print(f"\n{'='*60}")
print(f"All images saved to: {OUT}/")
print(f"{'='*60}")
print(f"\nSend these to professor:")
print(f"  1_frame74_intersections.jpg    - Lines & clustered intersections (H=None)")
print(f"  2_corner_selection.jpg         - Selected corners for homography")
print(f"  3_homography_reprojection.jpg  - Court projection via H^-1 (misaligned)")
print(f"  4_player_projection.jpg        - Player feet -> wrong court coordinates")
print(f"  5_multiframe_comparison.jpg    - Unstable detection across 4 frames")
print(f"  6_expected_vs_detected.jpg     - Expected court vs actual detections")
