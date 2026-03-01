"""Diagnostic: where is the net vs the far baseline?"""
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

# White detection on full mask
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
L = lab[:, :, 0]
L_court = cv2.bitwise_and(L, L, mask=court_mask)
court_pixels = L_court[court_mask > 0]
p88 = np.percentile(court_pixels, 88)
white_lab = (L_court > p88).astype(np.uint8) * 255
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
white_hsv = cv2.inRange(hsv, (0, 0, 140), (180, 90, 255))
white_mask = cv2.bitwise_or(white_lab, white_hsv)
white_mask = cv2.bitwise_and(white_mask, white_mask, mask=court_mask)
kernel = np.ones((3, 3), np.uint8)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

# Scan ALL horizontal rows for white pixel density
print("=== Horizontal white pixel scan (y=100 to y=800) ===")
vis = frame.copy()
for y in range(100, min(800, h)):
    row = white_mask[y, :]
    white_count = np.sum(row > 0)
    if white_count > 100:  # significant horizontal white
        print(f"  y={y}: {white_count} white pixels")
        # Highlight on vis
        vis[y, :, 1] = np.clip(vis[y, :, 1].astype(int) + 80, 0, 255).astype(np.uint8)

# Also check what's at y=253 (current KP0/KP1)
print(f"\n=== Frame at y=253 (current KP0/KP1) ===")
print(f"  White pixels at y=253: {np.sum(white_mask[253, :] > 0)}")
# Check brightness of the "net area" vs "court line"
# The net is a mesh/fabric — lower brightness, different texture
strip_250_260 = frame[250:260, :, :]
avg_brightness = np.mean(strip_250_260[white_mask[250:260, :] > 0]) if np.any(white_mask[250:260, :] > 0) else 0
print(f"  Avg brightness of white at y=250-260: {avg_brightness:.0f}")

# Check court mask boundary — where does the court START at top?
print(f"\n=== Court mask top boundary ===")
for y in range(0, h):
    if np.sum(court_mask[y, :]) > 10:
        print(f"  Court mask starts at y={y}")
        break

# Where is the net? The net is the white horizontal element 
# between the two kitchen lines
# Looking at current detection, kitchen lines at y≈292 and y≈432
# Net would be between those, around y≈250-260

# Let me look at the ACTUAL far baseline — scan top region of court
print(f"\n=== Searching for far baseline above net ===")
# The far baseline should be at the top edge of the court, 
# ABOVE the net (y < 250 roughly)
for y in range(100, 260):
    row = white_mask[y, :]
    wc = np.sum(row > 0)
    if wc > 50:
        cols = np.where(row > 0)[0]
        # Check if this is contiguous (line-like) vs scattered (net mesh)
        if len(cols) > 0:
            diffs = np.diff(cols)
            max_gap = np.max(diffs) if len(diffs) > 0 else 0
            n_segments = np.sum(diffs > 10) + 1
            print(f"  y={y}: {wc} white px, {n_segments} segments, max_gap={max_gap}")

cv2.imwrite(r'D:\Downloads\cv\tennis-pickleball-tracker\outputs\demo\net_baseline_debug.jpg', vis)
print("\nSaved net_baseline_debug.jpg")
