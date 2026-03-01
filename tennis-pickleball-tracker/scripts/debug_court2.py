import os, cv2, numpy as np, sys
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
sys.path.insert(0, r'D:\Downloads\cv\tennis-pickleball-tracker\src')
from ultralytics import YOLO

model = YOLO(r'D:\Downloads\cv\tennis-pickleball-tracker\models\pickleball_court\best.pt')
cap = cv2.VideoCapture(r'D:\Downloads\cv\tennis-pickleball-tracker\data\raw\pickleball_match.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

preds = model(frame, conf=0.3, verbose=False)
r = preds[0]
mask_data = r.masks[0]
# Check what xy actually is
print("type(mask_data.xy):", type(mask_data.xy))
print("len(mask_data.xy):", len(mask_data.xy))
xy0 = mask_data.xy[0]
print("xy[0] shape:", xy0.shape)
print("xy[0] min:", xy0.min(axis=0))
print("xy[0] max:", xy0.max(axis=0))
print("xy[0] first 5:", xy0[:5])
print()
print("Frame shape:", frame.shape)

# approxPolyDP
poly = xy0.reshape(-1, 1, 2).astype(np.float32)
peri = cv2.arcLength(poly, True)
print(f"Perimeter: {peri:.1f}")
for eps in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
    approx = cv2.approxPolyDP(poly, eps * peri, True)
    print(f"  eps={eps}: {len(approx)} points -> {approx.reshape(-1, 2)[:6]}")
