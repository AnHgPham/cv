"""Check what's REALLY at each y: white line or shadow?"""
import cv2, numpy as np, os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
from ultralytics import YOLO

model = YOLO(r'D:\Downloads\cv\tennis-pickleball-tracker\models\pickleball_court\best.pt')
cap = cv2.VideoCapture(r'D:\Downloads\cv\tennis-pickleball-tracker\data\raw\pickleball_match.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()
h, w = frame.shape[:2]

# Get mask
preds = model(frame, conf=0.3, verbose=False)
mask = cv2.resize(preds[0].masks[0].data[0].cpu().numpy(), (w, h))
mask = (mask > 0.5).astype(np.uint8)

# Check the ACTUAL pixel values at key y positions
print("=== Raw pixel analysis at horizontal line y positions ===")
print(f"  Court mask range: y={np.where(np.sum(mask, axis=1) > 0)[0][0]} to y={np.where(np.sum(mask, axis=1) > 0)[0][-1]}")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

for y in [219, 253, 292, 350, 380, 400, 432, 500, 530, 569, 600, 650, 713]:
    if y >= h:
        continue
    # Average brightness in the center strip (x=400 to x=1200)
    strip = gray[y, 400:1200]
    court_strip = mask[y, 400:1200]
    on_court = strip[court_strip > 0]
    avg = np.mean(on_court) if len(on_court) > 0 else 0
    
    # Count white pixels (>180) vs dark (<100)
    n_white = np.sum(on_court > 180) if len(on_court) > 0 else 0
    n_dark = np.sum(on_court < 100) if len(on_court) > 0 else 0
    n_total = len(on_court)
    
    # Determine nature
    if n_white > n_total * 0.3:
        nature = "WHITE LINE ✓"
    elif n_dark > n_total * 0.5:
        nature = "DARK/SHADOW ✗"
    else:
        nature = "court surface"
    
    print(f"  y={y}: avg_brightness={avg:.0f}, white={n_white}/{n_total}, dark={n_dark}/{n_total} → {nature}")

# Also show where the STRONGEST white horizontal lines are
print("\n=== Strongest white pixel rows (horizontal white lines) ===")
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
L = lab[:, :, 0]
L_court = cv2.bitwise_and(L, L, mask=mask)
court_top = np.where(np.sum(mask, axis=1) > 0)[0][0]
court_bot = np.where(np.sum(mask, axis=1) > 0)[0][-1]

# For each row, count pixels brighter than 200
row_white_counts = []
for y in range(court_top, court_bot+1):
    row = L_court[y, :]
    on_court = row[mask[y, :] > 0]
    if len(on_court) > 0:
        n_white = np.sum(on_court > 200)
        if n_white > 30:
            row_white_counts.append((y, n_white))

# Cluster the white rows
if row_white_counts:
    row_white_counts.sort(key=lambda x: x[0])
    clusters = [[row_white_counts[0]]]
    for item in row_white_counts[1:]:
        if item[0] - clusters[-1][-1][0] < 10:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    
    print(f"  Found {len(clusters)} white line clusters:")
    for i, cluster in enumerate(clusters):
        ys = [c[0] for c in cluster]
        counts = [c[1] for c in cluster]
        peak_y = ys[np.argmax(counts)]
        print(f"  Cluster {i}: y={min(ys)}-{max(ys)}, peak at y={peak_y} ({max(counts)} white px)")
