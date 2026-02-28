"""
Train YOLOv8-seg Court Detector for Pickleball
===============================================

Dataset: Pickleball Vision v1 (1848 images with Court polygon labels)
Output: Segmentation model that detects court boundary polygon
Post-process: Extract 4 corners from polygon → compute homography

Usage:
    python train_court_detector.py                    # Train
    python train_court_detector.py --epochs 30        # Fewer epochs
    python train_court_detector.py --test             # Test on samples
    python train_court_detector.py --test --weights path/to/best.pt
"""

import os
import yaml
import cv2
import numpy as np
import argparse


def prepare_court_only_dataset():
    """
    Create a filtered dataset with ONLY Court labels (class 1 → 0).
    Removes ball/-ball labels, remaps Court class to index 0.
    """
    src_base = "data/pickleball_vision_v1"
    dst_base = "data/pickleball_court_only"

    for split in ["train", "valid", "test"]:
        src_img = os.path.join(src_base, split, "images")
        src_lbl = os.path.join(src_base, split, "labels")
        dst_img = os.path.join(dst_base, split, "images")
        dst_lbl = os.path.join(dst_base, split, "labels")

        if not os.path.exists(src_lbl):
            continue

        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)

        count = 0
        for f in os.listdir(src_lbl):
            if not f.endswith(".txt"):
                continue

            with open(os.path.join(src_lbl, f)) as fh:
                lines = fh.readlines()

            court_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts[0] == "1":
                    court_lines.append("0 " + " ".join(parts[1:]) + "\n")

            if not court_lines:
                continue

            with open(os.path.join(dst_lbl, f), "w") as fh:
                fh.writelines(court_lines)

            img_name = f.replace(".txt", ".jpg")
            src_path = os.path.join(src_img, img_name)
            dst_path = os.path.join(dst_img, img_name)
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                import shutil
                shutil.copy2(src_path, dst_path)
            count += 1

        print(f"  {split}: {count} images with Court labels")

    data_yaml = {
        "names": ["Court"],
        "nc": 1,
        "train": os.path.abspath(os.path.join(dst_base, "train", "images")),
        "val": os.path.abspath(os.path.join(dst_base, "valid", "images")),
    }
    test_dir = os.path.join(dst_base, "test", "images")
    if os.path.exists(test_dir) and os.listdir(test_dir):
        data_yaml["test"] = os.path.abspath(test_dir)

    yaml_path = os.path.join(dst_base, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"  data.yaml: {yaml_path}")
    return yaml_path


def train(epochs=50, imgsz=640, batch=8, device="cpu"):
    """Train YOLOv8-seg on Court-only dataset."""
    from ultralytics import YOLO

    print("Preparing Court-only dataset...")
    yaml_path = prepare_court_only_dataset()

    model = YOLO("yolov8s-seg.pt")
    print(f"\nTraining YOLOv8s-seg for Court Detection")
    print(f"  Epochs: {epochs}, Batch: {batch}, ImgSz: {imgsz}, Device: {device}")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="models/pickleball_court",
        name="yolov8s_seg_court",
        exist_ok=True,
        patience=15,
        save=True,
        plots=True,
    )

    best = "models/pickleball_court/yolov8s_seg_court/weights/best.pt"
    print(f"\nDone! Best weights: {best}")
    return best


def extract_court_corners(mask_or_polygon, frame_shape):
    """
    Extract 4 court corners from a segmentation polygon or mask.

    The dataset labels courts with 4 polygon points in order:
    BL → BR → TR → TL. We reorder to TL, TR, BL, BR for
    homography correspondence with PICKLEBALL_COURT_CORNERS.
    """
    if isinstance(mask_or_polygon, np.ndarray) and mask_or_polygon.ndim == 2:
        contours, _ = cv2.findContours(
            mask_or_polygon.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        for eps in [0.02, 0.03, 0.05, 0.08]:
            approx = cv2.approxPolyDP(largest, eps * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                break
        else:
            pts = cv2.boxPoints(cv2.minAreaRect(largest)).astype(np.float32)
    else:
        pts = np.array(mask_or_polygon, dtype=np.float32)
        if len(pts) > 4:
            peri = cv2.arcLength(pts.reshape(-1, 1, 2).astype(np.float32), True)
            for eps in [0.02, 0.03, 0.05]:
                approx = cv2.approxPolyDP(
                    pts.reshape(-1, 1, 2).astype(np.float32), eps * peri, True
                )
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    break

    if len(pts) < 4:
        return None

    pts = pts[:4]
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 1] - pts[:, 0]
    tl = pts[int(np.argmin(sums))]
    br = pts[int(np.argmax(sums))]
    tr = pts[int(np.argmin(diffs))]
    bl = pts[int(np.argmax(diffs))]

    return np.array([tl, tr, bl, br], dtype=np.float32)


def test_detection(weights_path=None):
    """Test court detection and corner extraction on validation images."""
    from ultralytics import YOLO

    if weights_path is None:
        weights_path = "models/pickleball_court/yolov8s_seg_court/weights/best.pt"
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        print("Train first: python train_court_detector.py")
        return

    model = YOLO(weights_path)
    test_dir = "data/pickleball_court_only/valid/images"
    if not os.path.exists(test_dir):
        test_dir = "data/pickleball_vision_v1/valid/images"

    os.makedirs("outputs/court_detection_test", exist_ok=True)
    images = sorted(os.listdir(test_dir))[:10]

    for img_name in images:
        frame = cv2.imread(os.path.join(test_dir, img_name))
        if frame is None:
            continue

        results = model(frame, conf=0.3, verbose=False)

        for r in results:
            if r.masks is None:
                continue
            for j, (mask, box) in enumerate(zip(r.masks, r.boxes)):
                cls_name = r.names[int(box.cls[0])]
                if cls_name != "Court":
                    continue

                xy = r.masks.xy[j]
                corners = extract_court_corners(xy, frame.shape)

                if corners is not None:
                    poly = xy.astype(np.int32)
                    cv2.polylines(frame, [poly], True, (0, 255, 0), 2)

                    labels = ["TL", "TR", "BL", "BR"]
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
                    for i, ((cx, cy), lbl, col) in enumerate(
                        zip(corners, labels, colors)
                    ):
                        cv2.circle(frame, (int(cx), int(cy)), 8, col, -1)
                        cv2.putText(
                            frame, lbl, (int(cx) + 10, int(cy) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2,
                        )

                    conf = float(box.conf[0])
                    cv2.putText(
                        frame, f"Court {conf:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    )

        out = os.path.join("outputs/court_detection_test", f"detect_{img_name}")
        cv2.imwrite(out, frame)
        print(f"  {out}")

    print("Test done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    if args.test:
        test_detection(args.weights)
    else:
        train(args.epochs, args.imgsz, args.batch, args.device)
