"""
Module 2: Object Detection (Ball & Player)
==========================================

Three approaches implemented for comparison:

1. Feature Engineering (Baseline):
   - Viola-Jones (Haar cascade) + HOG + SVM for player detection
   - Color segmentation + Hough Circle for ball detection

2. Feature Learning (YOLO):
   - YOLOv8 fine-tuned for ball + player detection

3. Feature Learning (TrackNet):
   - Specialized heatmap-based network for small ball detection
   - Takes 3 consecutive frames as input to learn flying patterns

Knowledge applied:
- Viola-Jones / Haar-like features + AdaBoost cascade
- HOG (Histogram of Oriented Gradients) + SVM
- CNN (backbone of YOLO and TrackNet)
- YOLO (single-stage detection)
- Fast R-CNN / Faster R-CNN (two-stage detection)
- Feature Engineering vs Feature Learning comparison
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
import yaml

# Lazy imports for torch/torchvision (may crash on some Python versions)
HAS_TORCH = False
HAS_TORCHVISION = False
torch = None
nn = None
F = None

def _lazy_import_torch():
    """Lazily import torch to avoid crashes at module load time."""
    global HAS_TORCH, HAS_TORCHVISION, torch, nn, F
    if torch is not None:
        return
    try:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        torch = _torch
        nn = _nn
        F = _F
        HAS_TORCH = True
    except ImportError:
        return
    try:
        import torchvision as _tv
        HAS_TORCHVISION = True
    except (ImportError, OSError):
        pass


# Provide stub nn.Module base for class definitions at parse time
class _NNModuleStub:
    """Stub for nn.Module when torch is not available."""
    def __init__(self, *args, **kwargs):
        _lazy_import_torch()
        if HAS_TORCH:
            self.__class__.__bases__ = (nn.Module,)
            nn.Module.__init__(self)
        else:
            raise RuntimeError("This class requires PyTorch. Install: pip install torch")


# ============================================================================
# Data Structures
# ============================================================================

class Detection:
    """A single object detection result."""

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        class_id: int,
        class_name: str = "",
        center: Optional[Tuple[float, float]] = None,
    ):
        """
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            confidence: detection confidence [0, 1]
            class_id: class index (0=ball, 1=player)
            class_name: human-readable class name
            center: (cx, cy) center point (computed from bbox if not given)
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        if center is not None:
            self.center = center
        else:
            x1, y1, x2, y2 = bbox
            self.center = ((x1 + x2) / 2, (y1 + y2) / 2)

    def __repr__(self):
        return (
            f"Detection({self.class_name}, conf={self.confidence:.2f}, "
            f"bbox={self.bbox}, center=({self.center[0]:.1f}, {self.center[1]:.1f}))"
        )


# ============================================================================
# 1. Feature Engineering Baseline: Viola-Jones + HOG for Players
# ============================================================================

class ClassicalPlayerDetector:
    """
    Player detection using classical feature engineering methods.

    Method 1 - Viola-Jones (Haar Cascade):
        Uses Haar-like features (rectangular filters on integral image)
        with AdaBoost feature selection and cascade classifier for
        real-time detection. Originally designed for face detection,
        adapted here using person/upper-body cascades.

    Method 2 - HOG + SVM:
        Histogram of Oriented Gradients extracts gradient orientation
        distributions in local cells. Combined with a linear SVM
        classifier trained for pedestrian detection.
    """

    def __init__(self, method: str = "hog"):
        """
        Args:
            method: "haar" for Viola-Jones, "hog" for HOG+SVM
        """
        self.method = method

        if method == "haar":
            # Viola-Jones cascade classifier (full body)
            cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            if self.cascade.empty():
                # Fallback to upper body
                cascade_path = (
                    cv2.data.haarcascades + "haarcascade_upperbody.xml"
                )
                self.cascade = cv2.CascadeClassifier(cascade_path)

        elif method == "hog":
            # HOG descriptor + pre-trained SVM people detector
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(
                cv2.HOGDescriptor_getDefaultPeopleDetector()
            )

    def detect(
        self, frame: np.ndarray, confidence_threshold: float = 0.3
    ) -> List[Detection]:
        """
        Detect players in frame using classical methods.

        Returns list of Detection objects for players found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []

        if self.method == "haar":
            rects = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(40, 80),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for (x, y, w, h) in rects:
                det = Detection(
                    bbox=(x, y, x + w, y + h),
                    confidence=0.5,  # Haar doesn't give confidence
                    class_id=1,
                    class_name="player",
                )
                detections.append(det)

        elif self.method == "hog":
            rects, weights = self.hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05,
            )
            for (x, y, w, h), weight in zip(rects, weights):
                conf = float(weight[0]) if len(weight) > 0 else 0.5
                if conf >= confidence_threshold:
                    det = Detection(
                        bbox=(x, y, x + w, y + h),
                        confidence=min(conf, 1.0),
                        class_id=1,
                        class_name="player",
                    )
                    detections.append(det)

        return detections


class ClassicalBallDetector:
    """
    Ball detection using classical feature engineering methods.

    Uses color segmentation (isolate bright yellow/green ball) +
    Hough Circle Transform for circular shape detection.

    This is a baseline for comparison against deep learning methods.
    """

    def __init__(self):
        # Tennis ball color range in HSV (bright yellow/green)
        self.ball_lower_hsv = np.array([25, 80, 80])
        self.ball_upper_hsv = np.array([45, 255, 255])
        # Minimum and maximum radius (pixels)
        self.min_radius = 2
        self.max_radius = 20

    def detect(
        self, frame: np.ndarray, confidence_threshold: float = 0.3
    ) -> List[Detection]:
        """
        Detect ball using color segmentation + Hough Circles.

        Returns list of Detection objects (usually 0 or 1 ball).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.ball_lower_hsv, self.ball_upper_hsv)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Blur for Hough
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=20,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        detections = []
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            for (cx, cy, r) in circles:
                det = Detection(
                    bbox=(cx - r, cy - r, cx + r, cy + r),
                    confidence=0.5,
                    class_id=0,
                    class_name="ball",
                    center=(float(cx), float(cy)),
                )
                detections.append(det)

        return detections


# ============================================================================
# 2. YOLOv8 Detector (Feature Learning)
# ============================================================================

class YOLODetector:
    """
    YOLOv8-based object detector for ball and player detection.

    Uses Ultralytics YOLOv8 for single-stage detection.
    Can be fine-tuned on tennis/pickleball dataset.

    YOLO divides image into grid cells and predicts bounding boxes
    + class probabilities in a single forward pass.
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "0",
        classes: Optional[List[int]] = None,
    ):
        """
        Args:
            model_path: Path to YOLOv8 weights (.pt file)
            conf_threshold: Minimum confidence for detections
            iou_threshold: NMS IoU threshold
            device: CUDA device ("0") or "cpu"
            classes: List of class indices to detect (None = all)
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            raise ImportError(
                "Please install ultralytics: pip install ultralytics"
            )

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes

        # Class mapping for custom-trained model
        self.class_names = {0: "ball", 1: "player"}

    def detect(
        self, frame: np.ndarray, verbose: bool = False
    ) -> List[Detection]:
        """
        Run YOLOv8 inference on a single frame.

        Returns list of Detection objects.
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=self.classes,
            verbose=verbose,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                cls_name = self.class_names.get(
                    cls_id, result.names.get(cls_id, str(cls_id))
                )

                det = Detection(
                    bbox=tuple(bbox),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                )
                detections.append(det)

        return detections

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        name: str = "tennis_yolo",
    ):
        """
        Fine-tune YOLOv8 on custom dataset.

        Args:
            data_yaml: Path to dataset YAML config
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            name: Experiment name
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            patience=20,
            save=True,
            project="outputs/yolo",
        )


# ============================================================================
# 3. TrackNet (Specialized Ball Detection via Heatmap)
# ============================================================================

class TrackNetEncoder(_NNModuleStub):
    """
    TrackNet Encoder: VGG16-like architecture.

    Takes 3 consecutive frames (9 channels) as input.
    Progressively downsamples with conv + pooling layers.
    """

    def __init__(self, input_channels: int = 9):
        super().__init__()

        # Block 1: 9 -> 64
        self.conv1_1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 64 -> 128
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 128 -> 256
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4: 256 -> 512
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.bn4_3 = nn.BatchNorm2d(512)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool3(x)

        # Block 4 (no pooling - keep resolution for decoder)
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))

        return x


class TrackNetDecoder(_NNModuleStub):
    """
    TrackNet Decoder: Deconvolution layers to upsample back.

    Progressively upsamples feature maps back to input resolution
    to produce a heatmap indicating ball location.
    """

    def __init__(self, output_channels: int = 1):
        super().__init__()

        # Upsample block 1: 512 -> 256
        self.deconv1_1 = nn.ConvTranspose2d(512, 256, 3, padding=1)
        self.deconv1_2 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.deconv1_3 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Upsample block 2: 256 -> 128
        self.deconv2_1 = nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.deconv2_2 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Upsample block 3: 128 -> 64
        self.deconv3_1 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.deconv3_2 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Output: 64 -> 1 (heatmap)
        self.output_conv = nn.Conv2d(64, output_channels, 1)

        self.bn1_1 = nn.BatchNorm2d(256)
        self.bn1_2 = nn.BatchNorm2d(256)
        self.bn1_3 = nn.BatchNorm2d(256)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.bn3_2 = nn.BatchNorm2d(64)

    def forward(self, x):
        # Upsample block 1
        x = F.relu(self.bn1_1(self.deconv1_1(x)))
        x = F.relu(self.bn1_2(self.deconv1_2(x)))
        x = F.relu(self.bn1_3(self.deconv1_3(x)))
        x = self.up1(x)

        # Upsample block 2
        x = F.relu(self.bn2_1(self.deconv2_1(x)))
        x = F.relu(self.bn2_2(self.deconv2_2(x)))
        x = self.up2(x)

        # Upsample block 3
        x = F.relu(self.bn3_1(self.deconv3_1(x)))
        x = F.relu(self.bn3_2(self.deconv3_2(x)))
        x = self.up3(x)

        # Output heatmap
        x = torch.sigmoid(self.output_conv(x))

        return x


class TrackNet(_NNModuleStub):
    """
    TrackNet: Deep learning model for high-speed small ball detection.

    Architecture:
        - Encoder: VGG16-like CNN (9ch input from 3 consecutive frames)
        - Decoder: Deconvolution layers producing a heatmap
        - Output: Probability heatmap (H, W) of ball location

    The key insight is using 3 consecutive frames to learn
    temporal patterns (ball trajectory / flying patterns), enabling
    detection even when the ball is blurry from motion.

    Reference:
        Huang et al. (2019) "TrackNet: A Deep Learning Network for
        Tracking High-speed and Tiny Objects in Sports Applications"
    """

    def __init__(self, input_frames: int = 3, output_channels: int = 1):
        super().__init__()
        self.encoder = TrackNetEncoder(input_channels=input_frames * 3)
        self.decoder = TrackNetDecoder(output_channels=output_channels)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, input_frames*3, H, W) - concatenated consecutive frames

        Returns:
            (B, 1, H, W) heatmap of ball probability
        """
        features = self.encoder(x)
        heatmap = self.decoder(features)
        return heatmap


class TrackNetDetector:
    """
    TrackNet-based ball detector with pre/post-processing.

    Manages the frame buffer (3 consecutive frames) and
    extracts ball position from the output heatmap.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        input_size: Tuple[int, int] = (640, 360),
        device: str = "cuda",
        heatmap_threshold: float = 0.5,
        sigma: float = 2.5,
    ):
        _lazy_import_torch()
        if not HAS_TORCH:
            raise RuntimeError("TrackNetDetector requires PyTorch.")
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.input_size = input_size  # (W, H)
        self.heatmap_threshold = heatmap_threshold
        self.sigma = sigma

        self.model = TrackNet(input_frames=3, output_channels=1).to(self.device)

        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)

        self.model.eval()

        # Frame buffer: stores last 3 frames
        self.frame_buffer: List[np.ndarray] = []

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize and convert frame to RGB normalized array."""
        resized = cv2.resize(frame, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    def update_buffer(self, frame: np.ndarray):
        """Add frame to the rolling buffer of 3 frames."""
        processed = self.preprocess_frame(frame)
        self.frame_buffer.append(processed)
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)

    def create_input_tensor(self):
        """
        Create the 9-channel input tensor from 3 buffered frames.

        Returns None if buffer doesn't have 3 frames yet.
        """
        if len(self.frame_buffer) < 3:
            return None

        # Concatenate 3 frames along channel dimension: (H, W, 9)
        combined = np.concatenate(self.frame_buffer, axis=2)
        # Convert to (1, 9, H, W) tensor
        tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect ball in current frame using TrackNet.

        Adds frame to buffer, runs inference if 3 frames available,
        extracts ball position from heatmap.

        Returns list of Detection objects (0 or 1 ball).
        """
        original_h, original_w = frame.shape[:2]
        self.update_buffer(frame)

        input_tensor = self.create_input_tensor()
        if input_tensor is None:
            return []  # Not enough frames in buffer yet

        with torch.no_grad():
            heatmap = self.model(input_tensor)  # (1, 1, H, W)
        heatmap = heatmap[0, 0].cpu().numpy()  # (H, W)

        # Find peak in heatmap
        detections = self._extract_ball_from_heatmap(
            heatmap, original_w, original_h
        )
        return detections

    def _extract_ball_from_heatmap(
        self,
        heatmap: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> List[Detection]:
        """
        Extract ball position from the output heatmap.

        Thresholds the heatmap, finds contours, and returns
        the position of the largest blob above threshold.
        """
        # Threshold
        binary = (heatmap > self.heatmap_threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        # Find the largest contour (most likely the ball)
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)

        if M["m00"] == 0:
            return []

        # Centroid in heatmap coordinates
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Scale back to original frame coordinates
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        cx_orig = cx * scale_x
        cy_orig = cy * scale_y

        # Approximate bounding box (ball is small)
        r = 6  # approximate radius in pixels
        det = Detection(
            bbox=(
                int(cx_orig - r),
                int(cy_orig - r),
                int(cx_orig + r),
                int(cy_orig + r),
            ),
            confidence=float(np.max(heatmap)),
            class_id=0,
            class_name="ball",
            center=(cx_orig, cy_orig),
        )
        return [det]

    @staticmethod
    def generate_heatmap(
        x: float, y: float, width: int, height: int, sigma: float = 2.5
    ) -> np.ndarray:
        """
        Generate a Gaussian heatmap for training ground truth.

        Creates a 2D Gaussian centered at (x, y) with given sigma.
        Used to create target heatmaps from ball coordinate labels.

        Args:
            x, y: Ball center coordinates
            width, height: Heatmap dimensions
            sigma: Gaussian standard deviation

        Returns:
            (height, width) heatmap array with values in [0, 1]
        """
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        return heatmap.astype(np.float32)


# ============================================================================
# 4. Faster R-CNN Detector (Two-Stage, for Player Detection)
# ============================================================================

class FasterRCNNDetector:
    """
    Faster R-CNN for player detection.

    Two-stage detector:
    1. Region Proposal Network (RPN) proposes candidate regions
    2. Fast R-CNN head classifies and refines bounding boxes

    Uses pretrained torchvision Faster R-CNN (COCO) to detect "person" class.
    """

    def __init__(
        self,
        conf_threshold: float = 0.5,
        device: str = "cuda",
        person_class_id: int = 1,  # COCO person class
    ):
        _lazy_import_torch()
        import torchvision

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.conf_threshold = conf_threshold
        self.person_class_id = person_class_id

        # Load pretrained Faster R-CNN
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect players using Faster R-CNN.

        Filters results to only include "person" class detections.
        """
        # Preprocess: BGR -> RGB, normalize to [0, 1], to tensor
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            predictions = self.model([tensor])[0]

        detections = []
        for i in range(len(predictions["boxes"])):
            score = float(predictions["scores"][i])
            label = int(predictions["labels"][i])

            if label == self.person_class_id and score >= self.conf_threshold:
                bbox = predictions["boxes"][i].cpu().numpy().astype(int)
                det = Detection(
                    bbox=tuple(bbox),
                    confidence=score,
                    class_id=1,
                    class_name="player",
                )
                detections.append(det)

        return detections


# ============================================================================
# Focal Loss for TrackNet Training
# ============================================================================

class FocalLoss(_NNModuleStub):
    """
    Focal Loss for handling class imbalance in heatmap prediction.

    The ball occupies very few pixels compared to background.
    Focal loss down-weights easy (background) examples and focuses
    on hard (ball) examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()


# ============================================================================
# Utility: Non-Maximum Suppression
# ============================================================================

def non_max_suppression(
    detections: List[Detection], iou_threshold: float = 0.4
) -> List[Detection]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.

    Keeps the detection with highest confidence when two detections
    overlap more than iou_threshold.
    """
    if not detections:
        return []

    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)

        remaining = []
        for det in detections:
            if _compute_iou(best.bbox, det.bbox) < iou_threshold:
                remaining.append(det)
        detections = remaining

    return keep


def _compute_iou(
    box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union
