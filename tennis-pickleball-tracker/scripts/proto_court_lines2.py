"""Better hybrid court line detection using edge-based approach within seg mask."""
import os, cv2, numpy as np
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
from ultralytics import YOLO

model = YOLO(r'D:\Downloads\cv\tennis-pickleball-tracker\models\pickleball_court\best.pt')
cap = cv2.VideoCapture(r'D:\Downloads\cv\tennis-pickleball-tracker\data\raw\pickleball_match.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]

# Step 1: Get segmentation mask
preds = model(frame, conf=0.3, verbose=False)
mask_data = preds[0].masks[0]
mask_np = mask_data.data[0].cpu().numpy()
mask_resized = cv2.resize(mask_np, (w, h))
court_mask = (mask_resized > 0.5).astype(np.uint8)

# Step 2: Within court mask, detect edges
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Apply court mask
gray_court = cv2.bitwise_and(gray, gray, mask=court_mask)

# Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray_court, (5, 5), 1.5)

# Adaptive threshold to find bright lines on dark blue
# Or use Canny directly with tight thresholds
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# Mask edges to court region only
edges = cv2.bitwise_and(edges, edges, mask=court_mask)

# Dilate edges slightly
kernel = np.ones((2, 2), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)

cv2.imwrite(r'D:\Downloads\cv\tennis-pickleball-tracker\outputs\demo\edges_in_court.jpg', edges)

# Step 3: Hough Lines (use standard HoughLines for infinite lines, better for perspective)
lines_p = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                           minLineLength=60, maxLineGap=20)

print(f"HoughLinesP lines: {len(lines_p) if lines_p is not None else 0}")

# Also try standard Hough for comparison
lines_std = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)
print(f"HoughLines (standard): {len(lines_std) if lines_std is not None else 0}")

# Draw HoughLinesP
vis = frame.copy()
if lines_p is not None:
    for line in lines_p:
        x1, y1, x2, y2 = line[0]
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite(r'D:\Downloads\cv\tennis-pickleball-tracker\outputs\demo\edges_houghP.jpg', vis)

# Draw standard Hough lines (infinite lines)
vis2 = frame.copy()
if lines_std is not None:
    for line in lines_std[:20]:  # top 20
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        cv2.line(vis2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(f"\nTop 10 standard Hough lines (rho, theta_deg):")
    for line in lines_std[:10]:
        rho, theta = line[0]
        print(f"  rho={rho:.0f}, theta={np.degrees(theta):.1f}")
cv2.imwrite(r'D:\Downloads\cv\tennis-pickleball-tracker\outputs\demo\edges_hough_std.jpg', vis2)

# Try approach: color-based white line detection with better thresholds
# On blue court, white lines have high brightness and contrast to blue
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# Broader white detection
white1 = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
# Also detect via Lab color space (L channel = lightness)
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
L = lab[:, :, 0]
# White lines are brightest within court
L_court = cv2.bitwise_and(L, L, mask=court_mask)
# Threshold at high percentile within court
court_pixels = L_court[court_mask > 0]
p90 = np.percentile(court_pixels, 90)
p95 = np.percentile(court_pixels, 95)
print(f"\nLab L channel percentiles in court: p90={p90:.0f}, p95={p95:.0f}")
white_lab = (L_court > p90).astype(np.uint8) * 255

# Combine masks
white_combined = cv2.bitwise_or(white1, white_lab)
white_combined = cv2.bitwise_and(white_combined, white_combined, mask=court_mask)

# Morphological cleanup - thin lines
kernel_v = np.ones((1, 5), np.uint8)
kernel_h = np.ones((5, 1), np.uint8)
white_combined = cv2.morphologyEx(white_combined, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
white_combined = cv2.morphologyEx(white_combined, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

cv2.imwrite(r'D:\Downloads\cv\tennis-pickleball-tracker\outputs\demo\white_lines_v2.jpg', white_combined)

# Hough on this
edges2 = cv2.Canny(white_combined, 50, 150)
lines_p2 = cv2.HoughLinesP(edges2, rho=1, theta=np.pi/180, threshold=50,
                             minLineLength=80, maxLineGap=30)
print(f"\nHoughLinesP on white_v2: {len(lines_p2) if lines_p2 is not None else 0}")

vis3 = frame.copy()
if lines_p2 is not None:
    for line in lines_p2:
        x1, y1, x2, y2 = line[0]
        cv2.line(vis3, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite(r'D:\Downloads\cv\tennis-pickleball-tracker\outputs\demo\white_v2_hough.jpg', vis3)

print("Done! Check outputs/demo/ for all visualizations")
