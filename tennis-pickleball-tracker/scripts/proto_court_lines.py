"""Prototype hybrid court line detection: seg mask + white line filtering + Hough."""
import os, cv2, numpy as np, sys
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
sys.path.insert(0, r'D:\Downloads\cv\tennis-pickleball-tracker\src')

from ultralytics import YOLO

model = YOLO(r'D:\Downloads\cv\tennis-pickleball-tracker\models\pickleball_court\best.pt')
cap = cv2.VideoCapture(r'D:\Downloads\cv\tennis-pickleball-tracker\data\raw\pickleball_match.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]
print(f"Frame: {w}x{h}")

# Step 1: Get segmentation mask
preds = model(frame, conf=0.3, verbose=False)
r = preds[0]
mask_data = r.masks[0]
mask_np = mask_data.data[0].cpu().numpy()
mask_resized = cv2.resize(mask_np, (w, h))
court_mask = (mask_resized > 0.5).astype(np.uint8)
print(f"Court mask: {court_mask.sum()} pixels ({100*court_mask.sum()/(w*h):.1f}%)")

# Step 2: White line detection within court mask
# Convert to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# White lines: low saturation, high value
# Tune these thresholds
white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))

# Apply court mask to limit to court region
white_in_court = cv2.bitwise_and(white_mask, white_mask, mask=court_mask)

# Morphological cleanup
kernel = np.ones((3, 3), np.uint8)
white_in_court = cv2.morphologyEx(white_in_court, cv2.MORPH_CLOSE, kernel, iterations=2)
white_in_court = cv2.morphologyEx(white_in_court, cv2.MORPH_OPEN, kernel, iterations=1)

cv2.imwrite(r'D:\Downloads\cv\tennis-pickleball-tracker\outputs\demo\white_lines_mask.jpg', white_in_court)
print(f"White line pixels: {white_in_court.sum()//255}")

# Step 3: Edge detection on white lines
edges = cv2.Canny(white_in_court, 50, 150, apertureSize=3)

# Step 4: Hough Line Transform
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80,
                         minLineLength=100, maxLineGap=30)

print(f"Hough lines detected: {len(lines) if lines is not None else 0}")

# Draw all detected lines on frame
vis = frame.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite(r'D:\Downloads\cv\tennis-pickleball-tracker\outputs\demo\hough_lines_raw.jpg', vis)

# Step 5: Classify lines (horizontal vs vertical)
horizontal = []
vertical = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        angle = np.degrees(np.arctan2(dy, dx))
        length = np.sqrt(dx**2 + dy**2)
        if angle < 30:  # More horizontal
            horizontal.append((x1, y1, x2, y2, length, angle))
        elif angle > 60:  # More vertical
            vertical.append((x1, y1, x2, y2, length, angle))
        # Lines between 30-60 are diagonal (perspective sidelines)
        else:
            # For pickleball broadcast view, sidelines appear as diagonals
            # Classify by slope direction
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope > 0:
                vertical.append((x1, y1, x2, y2, length, angle))
            else:
                vertical.append((x1, y1, x2, y2, length, angle))

print(f"Horizontal: {len(horizontal)}, Vertical/Diagonal: {len(vertical)}")

# Step 6: Merge similar lines (by midpoint position)
def merge_lines(line_list, axis='y', threshold=20):
    """Merge nearby lines by clustering their midpoints."""
    if not line_list:
        return []
    # Sort by midpoint on the given axis
    if axis == 'y':
        sorted_lines = sorted(line_list, key=lambda l: (l[1] + l[3]) / 2)
    else:
        sorted_lines = sorted(line_list, key=lambda l: (l[0] + l[2]) / 2)
    
    groups = []
    current_group = [sorted_lines[0]]
    for line in sorted_lines[1:]:
        if axis == 'y':
            mid_curr = (current_group[-1][1] + current_group[-1][3]) / 2
            mid_next = (line[1] + line[3]) / 2
        else:
            mid_curr = (current_group[-1][0] + current_group[-1][2]) / 2
            mid_next = (line[0] + line[2]) / 2
        
        if abs(mid_next - mid_curr) < threshold:
            current_group.append(line)
        else:
            groups.append(current_group)
            current_group = [line]
    groups.append(current_group)
    
    # For each group, pick the longest line
    merged = []
    for group in groups:
        best = max(group, key=lambda l: l[4])  # longest
        merged.append(best)
    return merged

h_merged = merge_lines(horizontal, axis='y', threshold=30)
v_merged = merge_lines(vertical, axis='x', threshold=30)
print(f"After merge: H={len(h_merged)}, V={len(v_merged)}")

# Draw classified lines
vis2 = frame.copy()
for x1, y1, x2, y2, length, angle in h_merged:
    cv2.line(vis2, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green for horizontal
    mid_y = (y1 + y2) // 2
    cv2.putText(vis2, f"H y={mid_y}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

for x1, y1, x2, y2, length, angle in v_merged:
    cv2.line(vis2, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for vertical
    mid_x = (x1 + x2) // 2
    cv2.putText(vis2, f"V x={mid_x}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

cv2.imwrite(r'D:\Downloads\cv\tennis-pickleball-tracker\outputs\demo\court_lines_classified.jpg', vis2)
print("Saved classified lines visualization")

# Print line details
print("\nHorizontal lines:")
for x1, y1, x2, y2, length, angle in h_merged:
    print(f"  y_mid={(y1+y2)//2}, length={length:.0f}, angle={angle:.1f}")
print("\nVertical/Diagonal lines:")
for x1, y1, x2, y2, length, angle in v_merged:
    print(f"  x_mid={(x1+x2)//2}, length={length:.0f}, angle={angle:.1f}")
