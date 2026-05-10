"""End-to-end padel shot detection, tracking, classification, and export."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch

from classify import classify_shots
from detect import DEFAULT_DEVICE, DEFAULT_MODEL_PATH, detect_video
from track import track_video


DEFAULT_OUTPUT_DIR = Path("outputs")
ANNOTATED_VIDEO_NAME = "annotated.mp4"
JSON_OUTPUT_NAME = "shots.json"
CSV_OUTPUT_NAME = "shots.csv"
PLAYER_COLOR = (0, 180, 0)
RACKET_COLOR = (255, 80, 0)
BALL_COLOR = (0, 0, 255)
CONTACT_TEXT_COLOR = (255, 255, 255)
HUD_BG_COLOR = (20, 20, 20)
BOX_THICKNESS = 2
BALL_RADIUS = 6
HUD_X = 14
HUD_Y = 30
HUD_HEIGHT = 36
HUD_FONT_SCALE = 0.65
SHOT_FONT_SCALE = 1.0
TEXT_THICKNESS = 2
FOURCC = "mp4v"
OUTPUT_FPS_FALLBACK = 30.0
SHOT_LABEL_VERTICAL_STEP = 34
SUMMARY_LABEL_WIDTH = 14
CUDA_DEVICE = "cuda:0"
CPU_DEVICE = "cpu"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="Run padel shot analytics on a match video.")
    parser.add_argument("--video", type=Path, required=True, help="Path to input match video.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON, CSV, and annotated video.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to YOLOv8 model weights.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Inference device, for example auto, cpu, 0, or cuda:0.")
    return parser.parse_args()


def resolve_device(requested_device: str = DEFAULT_DEVICE) -> str:
    """Select CUDA when available for auto mode, otherwise fall back to CPU."""
    if requested_device != DEFAULT_DEVICE:
        print(f"Using requested inference device: {requested_device}")
        return requested_device

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA available: using {CUDA_DEVICE} ({device_name})")
        return CUDA_DEVICE

    print(f"CUDA not available: using {CPU_DEVICE}")
    return CPU_DEVICE


def draw_bbox(frame: Any, bbox: list[float], color: tuple[int, int, int], label: str) -> None:
    """Draw one labeled bounding box on a frame."""
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, HUD_FONT_SCALE, color, TEXT_THICKNESS)


def shot_lookup_by_frame(shots: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Index shot events by frame for annotation."""
    lookup: dict[int, list[dict[str, Any]]] = {}
    for shot in shots:
        lookup.setdefault(int(shot["frame_idx"]), []).append(shot)
    return lookup


def draw_hud(frame: Any, counts: Counter[str]) -> None:
    """Draw a running shot counter in the top-left corner."""
    text = f"Forehand: {counts['FOREHAND']} | Backhand: {counts['BACKHAND']} | Smash: {counts['SMASH/SERVE']}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], HUD_HEIGHT), HUD_BG_COLOR, -1)
    cv2.putText(frame, text, (HUD_X, HUD_Y), cv2.FONT_HERSHEY_SIMPLEX, HUD_FONT_SCALE, CONTACT_TEXT_COLOR, TEXT_THICKNESS)


def draw_state(frame: Any, state: dict[str, Any]) -> None:
    """Draw tracked players, rackets, and ball on a video frame."""
    for player in state.get("players", []):
        draw_bbox(frame, player["bbox"], PLAYER_COLOR, f"Player {player.get('id', '?')}")
    for racket in state.get("rackets", []):
        label = f"Racket P{racket.get('player_id', '?')}"
        draw_bbox(frame, racket["bbox"], RACKET_COLOR, label)
    ball = state.get("ball")
    if ball:
        cx, cy = [int(round(value)) for value in ball["centroid"]]
        cv2.circle(frame, (cx, cy), BALL_RADIUS, BALL_COLOR, -1)


def draw_shot_labels(frame: Any, shots: list[dict[str, Any]], counts: Counter[str]) -> None:
    """Overlay shot labels on contact frames and update running counts."""
    y = HUD_HEIGHT + 36
    for shot in shots:
        counts.update([shot["shot_type"]])
        text = f"P{shot['player_id']} {shot['shot_type']} {shot['confidence']:.2f}"
        cv2.putText(frame, text, (HUD_X, y), cv2.FONT_HERSHEY_SIMPLEX, SHOT_FONT_SCALE, CONTACT_TEXT_COLOR, TEXT_THICKNESS + 1)
        y += SHOT_LABEL_VERTICAL_STEP


def render_annotated_video(
    video_path: Path,
    states: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Render annotated video with detections, shot labels, and shot counts."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video for annotation: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or OUTPUT_FPS_FALLBACK
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*FOURCC), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create annotated video: {output_path}")

    shots_by_frame = shot_lookup_by_frame(shots)
    counts: Counter[str] = Counter()
    frame_idx = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_idx < len(states):
            draw_state(frame, states[frame_idx])
        draw_shot_labels(frame, shots_by_frame.get(frame_idx, []), counts)
        draw_hud(frame, counts)
        writer.write(frame)
        frame_idx += 1

    capture.release()
    writer.release()


def write_outputs(shots: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    """Write shot events to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_OUTPUT_NAME
    csv_path = output_dir / CSV_OUTPUT_NAME
    json_path.write_text(json.dumps(shots, indent=2), encoding="utf-8")
    pd.DataFrame(shots, columns=["frame_idx", "timestamp_sec", "player_id", "shot_type", "confidence"]).to_csv(csv_path, index=False)
    return json_path, csv_path


def print_summary(shots: list[dict[str, Any]], json_path: Path, csv_path: Path, video_path: Path) -> None:
    """Print a concise summary table after pipeline completion."""
    counts = Counter(shot["shot_type"] for shot in shots)
    print("\nShot Summary")
    print("============")
    print(f"{'Shot Type':<{SUMMARY_LABEL_WIDTH}}Count")
    print(f"{'-' * SUMMARY_LABEL_WIDTH}-----")
    for shot_type in ("FOREHAND", "BACKHAND", "SMASH/SERVE", "UNKNOWN"):
        print(f"{shot_type:<{SUMMARY_LABEL_WIDTH}}{counts[shot_type]}")
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"Video: {video_path}")


def run_pipeline(
    video_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    device: str = DEFAULT_DEVICE,
) -> list[dict[str, Any]]:
    """Run detection, tracking, classification, export, and annotation."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    detections = detect_video(video_path, model_path=model_path, device=resolved_device)
    states = track_video(video_path, detections=detections, model_path=model_path, device=resolved_device)
    shots = classify_shots(states)
    json_path, csv_path = write_outputs(shots, output_dir)
    annotated_path = output_dir / ANNOTATED_VIDEO_NAME
    render_annotated_video(video_path, states, shots, annotated_path)
    print_summary(shots, json_path, csv_path, annotated_path)
    return shots


def main() -> None:
    """Execute the CLI entry point."""
    args = parse_args()
    run_pipeline(args.video, args.output_dir, args.model, args.device)


if __name__ == "__main__":
    main()
