"""Debug right-side keypoint accuracy issue."""
import os, cv2, numpy as np
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import sys
sys.path.insert(0, r'D:\Downloads\cv\tennis-pickleball-tracker\src')
from court_detection import detect_court_lines_hybrid, draw_court_lines_overlay
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

# Run hybrid detection
result = detect_court_lines_hybrid(frame, court_mask)

print("=== DETECTION RESULTS ===")
print(f"Total lines: {len(result['lines'])}")
print(f"Baselines: {len(result['baselines'])}")
print(f"Sidelines: {len(result['sidelines'])}")
print(f"Kitchen lines: {len(result['kitchen_lines'])}")
print(f"Keypoints: {len(result['keypoints'])}")
print(f"Corners: {result['corners']}")

# Print line details
print("\nBaselines:")
for i, l in enumerate(result['baselines']):
    print(f"  [{i}] pts={l['pts']}, angle={l['angle']:.1f}, len={l['length']:.0f}, mid={l['mid']}")

print("\nSidelines:")
for i, l in enumerate(result['sidelines']):
    print(f"  [{i}] pts={l['pts']}, angle={l['angle']:.1f}, len={l['length']:.0f}, mid={l['mid']}")

print("\nKitchen lines:")
for i, l in enumerate(result['kitchen_lines']):
    print(f"  [{i}] pts={l['pts']}, angle={l['angle']:.1f}, len={l['length']:.0f}, mid={l['mid']}")

print("\nKeypoints:")
for i, pt in enumerate(result['keypoints']):
    # Check if keypoint is on white line
    x, y = pt
    region = frame[max(0,y-3):y+4, max(0,x-3):x+4]
    avg_brightness = np.mean(region) if region.size > 0 else 0
    print(f"  [{i}] ({x}, {y}) brightness={avg_brightness:.0f}")

# Visualize with annotations
vis = draw_court_lines_overlay(frame, result)
# Label keypoints
for i, pt in enumerate(result['keypoints']):
    cv2.putText(vis, f"KP{i}", (pt[0]+10, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
cv2.imwrite(r'D:\Downloads\cv\tennis-pickleball-tracker\outputs\demo\keypoint_debug.jpg', vis)
print("\nSaved keypoint_debug.jpg")
