"""
Interactive tool to manually click 4 court corners.
Instructions: Click on the 4 corners in order: TL, TR, BR, BL
"""
import cv2
import numpy as np

points = []
frame = None

def mouse_callback(event, x, y, flags, param):
    global points, frame
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Corner {len(points)}: ({x}, {y})")
        
        # Draw the point
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(frame, f"{len(points)}", (x+10, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw polygon if we have 2+ points
        if len(points) >= 2:
            cv2.polylines(frame, [np.array(points)], False, (0, 255, 0), 2)
        
        # Close polygon when 4 points
        if len(points) == 4:
            cv2.polylines(frame, [np.array(points)], True, (0, 255, 0), 3)
            print("\n4 corners selected!")
            print(f"expected = np.array([")
            for i, (px, py) in enumerate(points):
                label = ["far-left", "far-right", "near-right", "near-left"][i]
                print(f"    [{px}, {py}],    # {label}")
            print("], dtype=np.int32)")
        
        cv2.imshow("Select Court Corners", frame)

# Load frame 74
cap = cv2.VideoCapture("data/raw/Video Project 4.mp4")
for i in range(75):
    ret, f = cap.read()
cap.release()

frame = f.copy()
cv2.namedWindow("Select Court Corners")
cv2.setMouseCallback("Select Court Corners", mouse_callback)

print("Instructions:")
print("Click on 4 court corners in order:")
print("  1. Far-left baseline corner (top-left)")
print("  2. Far-right baseline corner (top-right)")
print("  3. Near-right baseline corner (bottom-right)")
print("  4. Near-left baseline corner (bottom-left)")
print("Press 'r' to reset, 'q' to quit")

while True:
    cv2.imshow("Select Court Corners", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r'):  # Reset
        points = []
        cap = cv2.VideoCapture("data/raw/Video Project 4.mp4")
        for i in range(75):
            ret, f = cap.read()
        cap.release()
        frame = f.copy()
        print("\nReset! Click 4 corners again.")
    
    elif key == ord('q') or len(points) == 4:  # Quit
        break

cv2.destroyAllWindows()

if len(points) == 4:
    print("\n✓ Done! Copy the 'expected = ...' code above into your script.")
