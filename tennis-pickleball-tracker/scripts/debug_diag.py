"""Fix: find actual right sideline by checking which diagonal lines sit on white court lines."""
import os, cv2, numpy as np
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import sys
sys.path.insert(0, r'D:\Downloads\cv\tennis-pickleball-tracker\src')
from ultralytics import YOLO

model = YOLO(r'D:\Downloads\cv\tennis-pickleball-tracker\models\pickleball_court\best.pt')
cap = cv2.VideoCapture(r'D:\Downloads\cv\tennis-pickleball-tracker\data\raw\pickleball_match.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()
h, w = frame.shape[:2]

# Get seg mask
preds = model(frame, conf=0.3, verbose=False)
mask_np = preds[0].masks[0].data[0].cpu().numpy()
court_mask = cv2.resize(mask_np, (w, h))
court_mask = (court_mask > 0.5).astype(np.uint8)

# Erode
erode_kernel = np.ones((15, 15), np.uint8)
court_mask_e = cv2.erode(court_mask, erode_kernel, iterations=1)

# White detection
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
L = lab[:, :, 0]
L_court = cv2.bitwise_and(L, L, mask=court_mask_e)
court_pixels = L_court[court_mask_e > 0]
p90 = np.percentile(court_pixels, 90)
white_lab = (L_court > p90).astype(np.uint8) * 255
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
white_hsv = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
white_mask = cv2.bitwise_or(white_lab, white_hsv)
white_mask = cv2.bitwise_and(white_mask, white_mask, mask=court_mask_e)
kernel = np.ones((3, 3), np.uint8)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

# Hough
edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)
lines_p = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=80, maxLineGap=30)

# Validate on white
def validate_line_on_white(x1, y1, x2, y2, white_img, min_ratio=0.4):
    n_samples = max(int(np.sqrt((x2-x1)**2 + (y2-y1)**2) / 3), 5)
    xs = np.linspace(x1, x2, n_samples).astype(int)
    ys = np.linspace(y1, y2, n_samples).astype(int)
    xs = np.clip(xs, 0, white_img.shape[1] - 1)
    ys = np.clip(ys, 0, white_img.shape[0] - 1)
    on_white = sum(1 for x, y in zip(xs, ys) if white_img[y, x] > 0)
    return on_white / n_samples

# Show ALL diagonal lines with their white ratio
print("All detected lines (diagonal, angle >= 20):")
vis = frame.copy()
for i, line in enumerate(lines_p):
    x1, y1, x2, y2 = line[0]
    dx = x2 - x1
    dy = y2 - y1
    angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
    length = np.sqrt(dx**2 + dy**2)
    mid_x = (x1+x2)/2
    white_ratio = validate_line_on_white(x1, y1, x2, y2, white_mask)
    
    if angle >= 20:
        color = (0,255,0) if white_ratio >= 0.4 else (0,0,255)
        cv2.line(vis, (x1,y1), (x2,y2), color, 2)
        print(f"  [{i}] ({x1},{y1})-({x2},{y2}) angle={angle:.1f} len={length:.0f} mid_x={mid_x:.0f} white={white_ratio:.2f}")

cv2.imwrite(r'D:\Downloads\cv\tennis-pickleball-tracker\outputs\demo\diagonal_lines_debug.jpg', vis)

# Also check: where are the actual white pixels on the right side?
# Sample a horizontal strip at y=300 (mid court)
print("\nWhite mask at y=300 (horizontal scan):")
row = white_mask[300, :]
nonzero = np.where(row > 0)[0]
if len(nonzero) > 0:
    print(f"  White pixels x-range: {nonzero[0]} to {nonzero[-1]}")
    # Find clusters
    diffs = np.diff(nonzero)
    gaps = np.where(diffs > 10)[0]
    starts = [nonzero[0]] + [nonzero[g+1] for g in gaps]
    ends = [nonzero[g] for g in gaps] + [nonzero[-1]]
    for s, e in zip(starts, ends):
        print(f"    segment: x={s} to x={e} (width {e-s})")
