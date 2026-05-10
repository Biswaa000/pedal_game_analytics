"""ByteTrack wrapper and association helpers for padel detections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from detect import (
    DEFAULT_CONFIDENCE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_PATH,
    PERSON_CLASS_ID,
    TENNIS_RACKET_CLASS_ID,
    DetectionDict,
    bbox_centroid,
    bbox_to_dict,
    resolve_model_path,
)


TRACKER_CONFIG = "bytetrack.yaml"
CENTROID_MATCH_SCALE = 0.75
MIN_IOU_FOR_MATCH = 0.01
DEFAULT_PLAYER_ID_START = 1


TrackState = dict[str, Any]


def bbox_iou(first: list[float] | tuple[float, float, float, float], second: list[float] | tuple[float, float, float, float]) -> float:
    """Compute intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denominator = first_area + second_area - inter_area
    return float(inter_area / denominator) if denominator else 0.0


def centroid_distance(first: list[float], second: list[float]) -> float:
    """Compute Euclidean distance between two centroids."""
    return float(np.linalg.norm(np.array(first, dtype=float) - np.array(second, dtype=float)))


def attach_rackets_to_players(
    rackets: list[DetectionDict],
    players: list[DetectionDict],
) -> list[DetectionDict]:
    """Assign each racket to the nearest plausible player."""
    assigned: list[DetectionDict] = []
    for racket in rackets:
        best_player: DetectionDict | None = None
        best_score = float("inf")
        for player in players:
            player_bbox = player["bbox"]
            player_width = max(1.0, float(player_bbox[2] - player_bbox[0]))
            player_height = max(1.0, float(player_bbox[3] - player_bbox[1]))
            max_distance = max(player_width, player_height) * CENTROID_MATCH_SCALE
            iou = bbox_iou(racket["bbox"], player_bbox)
            distance = centroid_distance(racket["centroid"], player["centroid"])
            if iou >= MIN_IOU_FOR_MATCH or distance <= max_distance:
                score = distance - (iou * max_distance)
                if score < best_score:
                    best_score = score
                    best_player = player
        racket_with_owner = dict(racket)
        racket_with_owner["player_id"] = best_player.get("id") if best_player else None
        assigned.append(racket_with_owner)
    return assigned


def player_from_track_box(box: Any, fallback_id: int) -> DetectionDict:
    """Convert an Ultralytics track box into a player tracking dict."""
    bbox_values = box.xyxy[0].detach().cpu().numpy().astype(float)
    bbox = tuple(float(value) for value in bbox_values)
    track_id = fallback_id
    if box.id is not None:
        track_id = int(box.id.item())
    detection = bbox_to_dict(bbox, float(box.conf.item()), source="bytetrack", class_id=PERSON_CLASS_ID)
    detection["id"] = int(track_id)
    return detection


def merge_detection_rackets(
    tracked_players: list[DetectionDict],
    frame_detection: DetectionDict | None,
) -> tuple[list[DetectionDict], DetectionDict | None]:
    """Use frame detections to add rackets and fallback ball to ByteTrack output."""
    if frame_detection is None:
        return [], None
    rackets = attach_rackets_to_players(frame_detection.get("rackets", []), tracked_players)
    return rackets, frame_detection.get("ball")


def track_video(
    video_path: Path,
    detections: list[DetectionDict] | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    device: str = DEFAULT_DEVICE,
) -> list[TrackState]:
    """Run ByteTrack over the video and return per-frame tracking states."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()

    model = YOLO(resolve_model_path(model_path))
    track_kwargs: dict[str, Any] = {}
    if device != DEFAULT_DEVICE:
        track_kwargs["device"] = device
    stream = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker=TRACKER_CONFIG,
        classes=[PERSON_CLASS_ID, TENNIS_RACKET_CLASS_ID],
        conf=DEFAULT_CONFIDENCE,
        verbose=False,
        **track_kwargs,
    )

    states: list[TrackState] = []
    for frame_idx, result in enumerate(stream):
        players: list[DetectionDict] = []
        fallback_id = DEFAULT_PLAYER_ID_START
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls.item()) == PERSON_CLASS_ID:
                    players.append(player_from_track_box(box, fallback_id))
                    fallback_id += 1

        frame_detection = detections[frame_idx] if detections and frame_idx < len(detections) else None
        if not players and frame_detection is not None:
            players = [dict(player, id=index + DEFAULT_PLAYER_ID_START) for index, player in enumerate(frame_detection.get("players", []))]
        rackets, ball = merge_detection_rackets(players, frame_detection)
        states.append(
            {
                "frame_idx": int(frame_idx),
                "timestamp_sec": float(frame_idx / fps) if fps else 0.0,
                "players": players,
                "rackets": rackets,
                "ball": ball,
            }
        )

    return states
