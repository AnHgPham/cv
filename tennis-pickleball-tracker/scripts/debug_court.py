"""Debug why SegmentationCourtDetector fails despite model detecting court."""
import os, cv2, numpy as np, sys
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
sys.path.insert(0, r'D:\Downloads\cv\tennis-pickleball-tracker\src')
from court_detection import SegmentationCourtDetector, extract_court_corners_from_segmentation, PICKLEBALL_COURT_CORNERS

det = SegmentationCourtDetector(
    model_path=r'D:\Downloads\cv\tennis-pickleball-tracker\models\pickleball_court\best.pt',
    conf_threshold=0.3
)

cap = cv2.VideoCapture(r'D:\Downloads\cv\tennis-pickleball-tracker\data\raw\pickleball_match.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

# Step 1: Raw model prediction
model = det._get_model()
preds = model(frame, conf=0.3, verbose=False)

for r in preds:
    if r.masks is None:
        print("No masks!")
        continue
    for j, (mask_data, box) in enumerate(zip(r.masks, r.boxes)):
        cls_id = int(box.cls[0])
        cls_name = r.names.get(cls_id, str(cls_id))
        conf = float(box.conf[0])
        print(f"Detection: class={cls_name}, conf={conf:.3f}")
        
        # Check polygon access
        if hasattr(mask_data, 'xy'):
            xy = mask_data.xy[j] if j < len(mask_data.xy) else None
            if xy is not None:
                print(f"  Polygon points: {len(xy)}")
            else:
                print(f"  xy[{j}] is None!")
                # Try xy[0]
                xy = mask_data.xy[0] if len(mask_data.xy) > 0 else None
                print(f"  Trying xy[0]: {len(xy) if xy is not None else 'None'} points")
        else:
            print("  No xy attribute!")
        
        # Try extract corners
        if xy is not None and len(xy) > 0:
            print(f"  Polygon shape: {xy.shape}")
            corners = extract_court_corners_from_segmentation(xy, frame.shape)
            if corners is None:
                print("  extract_court_corners_from_segmentation returned None!")
            else:
                print(f"  Corners: {corners}")
                # Try homography
                H, _ = cv2.findHomography(
                    corners[:4].astype(np.float32),
                    PICKLEBALL_COURT_CORNERS[:4].astype(np.float32),
                    cv2.RANSAC, 5.0,
                )
                if H is None:
                    print("  findHomography returned None!")
                else:
                    print(f"  Homography computed OK")
                    print(f"  H = {H}")

# Step 2: Full pipeline call
print("\n--- Full detect_and_compute_homography ---")
H, result = det.detect_and_compute_homography(frame)
print(f"  H = {'OK' if H is not None else 'None'}")
print(f"  Result: {result}")
