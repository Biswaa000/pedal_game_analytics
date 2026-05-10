"""YOLO and OpenCV detection utilities for padel footage."""

from __future__ import annotations

from pathlib import Path
from typing import Any   #flexible type hints

import cv2
import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32
TENNIS_RACKET_CLASS_ID = 38
DEFAULT_MODEL_NAME = "yolov8m.pt"
DEFAULT_MODEL_PATH = Path("models") / DEFAULT_MODEL_NAME
DEFAULT_DEVICE = "auto"
DEFAULT_CONFIDENCE = 0.25  #ignore weak detections below 25%
DEFAULT_IOU = 0.45         #non-max suppression threshold to filter overlapping boxes
MOG2_HISTORY = 120
MOG2_VAR_THRESHOLD = 24    #Controls motion sensitivity.
MORPH_KERNEL_SIZE = 3
MIN_BLOB_AREA = 8
MAX_BLOB_AREA = 450
MIN_CIRCULARITY = 0.2
BALL_PADDING = 6           # visibility of ball in bounding box, in pixels


BBox = tuple[float, float, float, float]  #left top, right bottom
DetectionDict = dict[str, Any]            #any data type allowed
   

def resolve_model_path(model_path: Path = DEFAULT_MODEL_PATH) -> str:
    """Return a local model path when present, otherwise a YOLO model name."""
    return str(model_path) if model_path.exists() else DEFAULT_MODEL_NAME


def bbox_centroid(bbox: BBox) -> tuple[float, float]:
    """Compute the center point for an xyxy bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_to_dict(
    bbox: BBox,
    confidence: float,
    source: str,
    class_id: int | None = None,
) -> DetectionDict:
    """Convert a bounding box into the structured detection format."""
    cx, cy = bbox_centroid(bbox)
    data: DetectionDict = {
        "bbox": [float(v) for v in bbox],
        "centroid": [float(cx), float(cy)],
        "confidence": float(confidence),
        "source": source,
    }
    if class_id is not None:
        data["class_id"] = int(class_id)
    return data


def create_ball_blob_detector() -> cv2.SimpleBlobDetector:
    """Create a small, permissive blob detector for fast ball candidates."""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = float(MIN_BLOB_AREA)
    params.maxArea = float(MAX_BLOB_AREA)
    params.filterByCircularity = True
    params.minCircularity = float(MIN_CIRCULARITY)
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = False
    return cv2.SimpleBlobDetector_create(params)


def detect_ball_with_background_subtraction(
    frame: np.ndarray,
    subtractor: cv2.BackgroundSubtractor,
    blob_detector: cv2.SimpleBlobDetector,
) -> DetectionDict | None:
    """Detect a candidate ball using foreground motion and blob shape."""
    foreground = subtractor.apply(frame)
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_DILATE, kernel)
    keypoints = blob_detector.detect(foreground)
    if not keypoints:
        return None

    keypoint = max(keypoints, key=lambda item: item.response if item.response else item.size)
    cx, cy = keypoint.pt
    radius = max(keypoint.size / 2.0, float(BALL_PADDING))
    height, width = frame.shape[:2]
    bbox = (
        max(0.0, cx - radius),
        max(0.0, cy - radius),
        min(float(width - 1), cx + radius),
        min(float(height - 1), cy + radius),
    )
    return bbox_to_dict(bbox, confidence=0.35, source="background_subtraction")


def extract_yolo_detections(result: Any) -> tuple[list[DetectionDict], list[DetectionDict], DetectionDict | None]:
    """Extract player, racket, and best ball detections from a YOLO result."""
    players: list[DetectionDict] = []
    rackets: list[DetectionDict] = []
    balls: list[DetectionDict] = []
    if result.boxes is None:
        return players, rackets, None

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        bbox_values = box.xyxy[0].detach().cpu().numpy().astype(float)
        bbox = tuple(float(value) for value in bbox_values)
        detection = bbox_to_dict(bbox, confidence, source="yolo", class_id=class_id)
        if class_id == PERSON_CLASS_ID:
            players.append(detection)
        elif class_id == TENNIS_RACKET_CLASS_ID:
            rackets.append(detection)
        elif class_id == SPORTS_BALL_CLASS_ID:
            balls.append(detection)

    best_ball = max(balls, key=lambda item: float(item["confidence"])) if balls else None
    return players, rackets, best_ball


def detect_frame(
    model: YOLO,
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    subtractor: cv2.BackgroundSubtractor,
    blob_detector: cv2.SimpleBlobDetector,
    device: str = DEFAULT_DEVICE,
) -> DetectionDict:
    """Run YOLO and ball fallback detection on a single frame."""
    predict_kwargs: dict[str, Any] = {}
    if device != DEFAULT_DEVICE:
        predict_kwargs["device"] = device
    results = model.predict(frame, conf=DEFAULT_CONFIDENCE, iou=DEFAULT_IOU, verbose=False, **predict_kwargs)
    players, rackets, ball = extract_yolo_detections(results[0])
    if ball is None:
        ball = detect_ball_with_background_subtraction(frame, subtractor, blob_detector)

    return {
        "frame_idx": int(frame_idx),
        "timestamp_sec": float(frame_idx / fps) if fps else 0.0,
        "players": players,
        "rackets": rackets,
        "ball": ball,
    }


def detect_video(
    video_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    device: str = DEFAULT_DEVICE,
) -> list[DetectionDict]:
    """Run frame-by-frame detection over a video and return structured detections."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    model = YOLO(resolve_model_path(model_path))
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=False,
    )
    blob_detector = create_ball_blob_detector()

    detections: list[DetectionDict] = []
    frame_idx = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        detections.append(detect_frame(model, frame, frame_idx, fps, subtractor, blob_detector, device=device))
        frame_idx += 1

    capture.release()
    return detections
