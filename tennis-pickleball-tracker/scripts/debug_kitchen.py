"""Quick diagnostic: what kitchen lines are detected?"""
import os, cv2, numpy as np
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import sys
sys.path.insert(0, r'D:\Downloads\cv\tennis-pickleball-tracker\src')
from ultralytics import YOLO
from court_detection import detect_court_lines_hybrid

model = YOLO(r'D:\Downloads\cv\tennis-pickleball-tracker\models\pickleball_court\best.pt')
cap = cv2.VideoCapture(r'D:\Downloads\cv\tennis-pickleball-tracker\data\raw\pickleball_match.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()
h, w = frame.shape[:2]

preds = model(frame, conf=0.3, verbose=False)
mask_np = preds[0].masks[0].data[0].cpu().numpy()
court_mask = cv2.resize(mask_np, (w, h))
court_mask = (court_mask > 0.5).astype(np.uint8)

result = detect_court_lines_hybrid(frame, court_mask)

print("=== Baselines ===")
for i, bl in enumerate(result["baselines"]):
    print(f"  [{i}] y={bl['mid'][1]:.0f} len={bl['length']:.0f}")

print("\n=== Kitchen lines ===")
for i, kl in enumerate(result["kitchen_lines"]):
    print(f"  [{i}] y={kl['mid'][1]:.0f} len={kl['length']:.0f}")

print(f"\n=== Net ===")
nl = result.get("net_line")
if nl:
    print(f"  y={nl['mid'][1]:.0f} len={nl['length']:.0f}")
else:
    print("  None (not detected!)")

print(f"\n=== Center line ===")
cl = result.get("center_line")
if cl:
    print(f"  mid_x={cl['mid'][0]:.0f} angle={cl['angle']:.1f}")
else:
    print("  None")

print(f"\n=== All Hough horizontal lines ===")
horiz = [l for l in result["lines"] if l["angle"] < 20]
for l in sorted(horiz, key=lambda l: l["mid"][1]):
    print(f"  y={l['mid'][1]:.0f} len={l['length']:.0f}")
