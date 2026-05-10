"""Rule-based shot classification from tracked padel geometry."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


CONTACT_DISTANCE_PX = 55.0
MIN_SHOT_GAP_FRAMES = 15
VELOCITY_WINDOW_FRAMES = 3
SMASH_VERTICAL_FACTOR = -0.2
UPPER_BODY_FACTOR = 0.3
SIDE_FACTOR = 0.1
MID_HEIGHT_FACTOR = 0.2
MAX_CONFIDENCE = 1.0
MIN_CONFIDENCE = 0.0
BASE_CONTACT_CONFIDENCE = 0.35
GEOMETRY_CONFIDENCE_WEIGHT = 0.45
VELOCITY_CONFIDENCE_WEIGHT = 0.2
VELOCITY_NORMALIZER = 25.0


TrackState = dict[str, Any]
ShotEvent = dict[str, Any]


def distance(first: list[float], second: list[float]) -> float:
    """Compute Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(first, dtype=float) - np.array(second, dtype=float)))


def clamp(value: float, lower: float = MIN_CONFIDENCE, upper: float = MAX_CONFIDENCE) -> float:
    """Clamp a floating point value into a bounded range."""
    return max(lower, min(upper, value))


def find_player(players: list[dict[str, Any]], player_id: int | None) -> dict[str, Any] | None:
    """Find a player by id in a frame state."""
    if player_id is None:
        return None
    return next((player for player in players if player.get("id") == player_id), None)


def racket_velocity(states: list[TrackState], frame_idx: int, player_id: int | None) -> tuple[float, float]:
    """Estimate racket velocity using positions around the contact frame."""
    previous_centroid: list[float] | None = None
    next_centroid: list[float] | None = None
    start = max(0, frame_idx - VELOCITY_WINDOW_FRAMES)
    end = min(len(states) - 1, frame_idx + VELOCITY_WINDOW_FRAMES)

    for index in range(frame_idx, start - 1, -1):
        racket = next((item for item in states[index].get("rackets", []) if item.get("player_id") == player_id), None)
        if racket:
            previous_centroid = racket["centroid"]
            break

    for index in range(frame_idx, end + 1):
        racket = next((item for item in states[index].get("rackets", []) if item.get("player_id") == player_id), None)
        if racket:
            next_centroid = racket["centroid"]
            break

    if previous_centroid is None or next_centroid is None:
        return 0.0, 0.0
    return float(next_centroid[0] - previous_centroid[0]), float(next_centroid[1] - previous_centroid[1])


def classify_contact(
    state: TrackState,
    player: dict[str, Any],
    racket: dict[str, Any],
    velocity: tuple[float, float],
) -> tuple[str, float]:
    """Classify a shot at a contact frame with geometric rules."""
    player_bbox = player["bbox"]
    player_width = max(1.0, float(player_bbox[2] - player_bbox[0]))
    player_height = max(1.0, float(player_bbox[3] - player_bbox[1]))
    player_top = float(player_bbox[1])
    player_cx, player_cy = player["centroid"]
    racket_cx, racket_cy = racket["centroid"]
    relative_x = float(racket_cx - player_cx)
    relative_y = float(racket_cy - player_cy)
    velocity_strength = clamp(np.linalg.norm(np.array(velocity, dtype=float)) / VELOCITY_NORMALIZER)

    smash_margin = max(0.0, (-relative_y) - abs(SMASH_VERTICAL_FACTOR) * player_height)
    upper_body_limit = player_top + UPPER_BODY_FACTOR * player_height
    is_smash = relative_y < SMASH_VERTICAL_FACTOR * player_height and racket_cy < upper_body_limit
    if is_smash:
        geometry_strength = clamp(smash_margin / max(1.0, player_height * UPPER_BODY_FACTOR))
        return "SMASH/SERVE", clamp(BASE_CONTACT_CONFIDENCE + GEOMETRY_CONFIDENCE_WEIGHT * geometry_strength + VELOCITY_CONFIDENCE_WEIGHT * velocity_strength)

    is_mid_height = abs(relative_y) <= MID_HEIGHT_FACTOR * player_height
    forehand_margin = relative_x - SIDE_FACTOR * player_width
    if relative_x > SIDE_FACTOR * player_width and is_mid_height:
        geometry_strength = clamp(forehand_margin / max(1.0, player_width * SIDE_FACTOR))
        return "FOREHAND", clamp(BASE_CONTACT_CONFIDENCE + GEOMETRY_CONFIDENCE_WEIGHT * geometry_strength + VELOCITY_CONFIDENCE_WEIGHT * velocity_strength)

    backhand_margin = abs(relative_x) - SIDE_FACTOR * player_width
    if relative_x < -SIDE_FACTOR * player_width and is_mid_height:
        geometry_strength = clamp(backhand_margin / max(1.0, player_width * SIDE_FACTOR))
        return "BACKHAND", clamp(BASE_CONTACT_CONFIDENCE + GEOMETRY_CONFIDENCE_WEIGHT * geometry_strength + VELOCITY_CONFIDENCE_WEIGHT * velocity_strength)

    return "UNKNOWN", clamp(BASE_CONTACT_CONFIDENCE * 0.5 + VELOCITY_CONFIDENCE_WEIGHT * velocity_strength)


def contact_candidates(state: TrackState) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return player-racket pairs whose racket is near the detected ball."""
    ball = state.get("ball")
    if not ball:
        return []
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for racket in state.get("rackets", []):
        player = find_player(state.get("players", []), racket.get("player_id"))
        if player is None:
            continue
        if distance(ball["centroid"], racket["centroid"]) <= CONTACT_DISTANCE_PX:
            candidates.append((player, racket))
    return candidates


def classify_shots(states: list[TrackState]) -> list[ShotEvent]:
    """Classify shot events from tracked frame states."""
    shots: list[ShotEvent] = []
    last_shot_frame: dict[int, int] = defaultdict(lambda: -MIN_SHOT_GAP_FRAMES)
    for frame_idx, state in enumerate(states):
        for player, racket in contact_candidates(state):
            player_id = int(player["id"])
            if int(state["frame_idx"]) - last_shot_frame[player_id] < MIN_SHOT_GAP_FRAMES:
                continue
            velocity = racket_velocity(states, frame_idx, player_id)
            shot_type, confidence = classify_contact(state, player, racket, velocity)
            shots.append(
                {
                    "frame_idx": int(state["frame_idx"]),
                    "timestamp_sec": round(float(state.get("timestamp_sec", 0.0)), 3),
                    "player_id": player_id,
                    "shot_type": shot_type,
                    "confidence": round(float(confidence), 3),
                }
            )
            last_shot_frame[player_id] = int(state["frame_idx"])
    return shots

