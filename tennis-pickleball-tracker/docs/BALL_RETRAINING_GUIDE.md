# Quy trình Train lại Ball Detection cho Pickleball

## Tại sao cần train lại?
- Model hiện tại (YOLOv8) train trên ảnh Roboflow → chủ yếu **close-up, ảnh tĩnh**
- Video match thật: bóng **rất nhỏ** (~10-20px), blur, di chuyển nhanh
- Kết quả: nhiều **false positive** (detect nhầm vật thể khác là bóng)

## Bước 1: Trích frames từ video match
```bash
python -c "
from src.pipeline import DataPreprocessor
dp = DataPreprocessor()
dp.extract_frames('data/raw/pickleball_match.mp4', 'data/match_frames', frame_interval=5, max_frames=500)
"
```
→ Trích ~500 frames (mỗi 5 frame) từ video match

## Bước 2: Upload và label trên Roboflow
1. Vào [app.roboflow.com](https://app.roboflow.com)
2. Tạo project mới: "Pickleball Ball Detection"
3. Upload frames từ `data/match_frames/`
4. Label bóng: class `ball`
   - Kể cả bóng méo (motion blur)
   - Kể cả bóng mờ (xa camera)
   - Bounding box chặt sát bóng
5. Export → YOLOv8 format → download

## Bước 3: Merge datasets
```bash
# Copy labeled data vào data/merged/
# Merge data.yaml files
python scripts/merge_datasets.py \
    data/pickleball_vision_v9 \
    data/pickleball_ball_owl \
    data/match_labeled \
    --output data/merged_ball
```

## Bước 4: Train YOLOv8
```bash
yolo detect train \
    model=yolov8n.pt \
    data=data/merged_ball/data.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    name=pickleball_ball_v2
```

## Bước 5: Đánh giá và deploy
```bash
# Evaluate
yolo detect val model=runs/detect/pickleball_ball_v2/weights/best.pt data=data/merged_ball/data.yaml

# Copy model
cp runs/detect/pickleball_ball_v2/weights/best.pt models/pickleball_ball/best.pt
```

## Tips
- **Augmentations (Roboflow):** Flip, Rotate ±15°, Brightness ±20%, Blur, Mosaic
- **Negative samples:** Thêm ảnh KHÔNG có bóng để giảm false positive
- **Epochs:** 100-200 epochs, patience=50
- **Image size:** 640 hoặc 1280 (nếu GPU đủ VRAM)
