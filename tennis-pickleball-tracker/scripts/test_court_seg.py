import os, cv2, numpy as np
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
from ultralytics import YOLO

model = YOLO(r'D:\Downloads\cv\tennis-pickleball-tracker\models\pickleball_court\best.pt')
cap = cv2.VideoCapture(r'D:\Downloads\cv\tennis-pickleball-tracker\data\raw\pickleball_match.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

results = model(frame, conf=0.3, verbose=False)
for r in results:
    n_boxes = len(r.boxes) if r.boxes is not None else 0
    n_masks = len(r.masks) if r.masks is not None else 0
    print(f"Boxes: {n_boxes}, Masks: {n_masks}")
    if r.masks is not None:
        for j, (mask_data, box) in enumerate(zip(r.masks, r.boxes)):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = r.names.get(cls_id, str(cls_id))
            print(f"  [{j}] class={cls_name}, conf={conf:.3f}")
            if hasattr(mask_data, 'xy'):
                polys = mask_data.xy
                for k, poly in enumerate(polys):
                    print(f"    polygon[{k}]: {len(poly)} points")
            # Draw mask on frame
            mask_np = mask_data.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
            overlay = frame.copy()
            overlay[mask_resized > 0.5] = [0, 255, 0]
            frame_out = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            cv2.imwrite(r'D:\Downloads\cv\tennis-pickleball-tracker\outputs\demo\court_seg_test.jpg', frame_out)
            print("  Saved court_seg_test.jpg")
    else:
        print("No masks detected!")
