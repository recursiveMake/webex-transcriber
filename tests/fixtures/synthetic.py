"""Generate synthetic WebEx-like frames for unit testing.

Creates frames with participant tiles that have:
- A dark background
- Coloured avatar region
- Name text overlay at the bottom
- A green mic icon blob when the participant is active
"""

from __future__ import annotations

import cv2
import numpy as np


# WebEx active mic green (matches _GREEN_LOWER/_GREEN_UPPER in layout.py)
_MIC_GREEN = (60, 210, 80)   # BGR

# Default frame dimensions
FRAME_W, FRAME_H = 1280, 720


def make_participant_tile(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    name: str,
    active: bool = False,
    avatar_colour: tuple[int, int, int] = (80, 80, 120),
) -> None:
    """Draw a WebEx-style participant tile onto `frame` in place."""
    # Tile background
    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 1)

    # Avatar area (top 70%)
    av_h = int(h * 0.70)
    cv2.rectangle(frame, (x + 4, y + 4), (x + w - 4, y + av_h), avatar_colour, -1)

    # Name bar (bottom 30%)
    name_y = y + av_h
    name_h = h - av_h
    cv2.rectangle(frame, (x, name_y), (x + w, y + h), (20, 20, 20), -1)

    # Name text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, w / 300)
    text_size = cv2.getTextSize(name, font, font_scale, 1)[0]
    tx = x + (w - text_size[0]) // 2
    ty = name_y + (name_h + text_size[1]) // 2
    cv2.putText(frame, name, (tx, ty), font, font_scale, (220, 220, 220), 1, cv2.LINE_AA)

    # Active-speaker cues: green border around the tile + mic icon blob.
    # The border is the primary signal used by find_bordered_tiles.
    # The mic blob is positioned well inside the tile so it stays as a
    # separate contour from the border after morphological operations.
    if active:
        cv2.rectangle(frame, (x, y), (x + w, y + h), _MIC_GREEN, 2)
        mic_x = x + w - 18
        mic_y = y + h - 22
        cv2.circle(frame, (mic_x, mic_y), 6, _MIC_GREEN, -1)


def make_webex_frame(
    participants: list[dict],
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
) -> np.ndarray:
    """Build a synthetic WebEx-style frame.

    Args:
        participants: list of dicts with keys:
            - name (str)
            - active (bool): whether the mic is green
            - tile (tuple[int,int,int,int]): (x, y, w, h)
            - avatar_colour (optional tuple)

    Returns:
        BGR numpy array of shape (frame_h, frame_w, 3).
    """
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    frame[:] = (15, 15, 15)  # dark background

    for p in participants:
        x, y, w, h = p["tile"]
        make_participant_tile(
            frame, x, y, w, h,
            name=p["name"],
            active=p.get("active", False),
            avatar_colour=p.get("avatar_colour", (80, 80, 120)),
        )
    return frame


# ---------------------------------------------------------------------------
# Standard test scenarios
# ---------------------------------------------------------------------------

_TILE_W, _TILE_H = 200, 130
_RIGHT_X = FRAME_W - _TILE_W - 10


def two_participant_frame(
    active: str = "Alice",
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
) -> np.ndarray:
    """Frame with two participants in a right-side strip; one active."""
    participants = [
        {
            "name": "Alice",
            "tile": (_RIGHT_X, 10, _TILE_W, _TILE_H),
            "active": active == "Alice",
            "avatar_colour": (120, 60, 60),   # blue-ish, H≈120 — outside green range
        },
        {
            "name": "Bob",
            "tile": (_RIGHT_X, 10 + _TILE_H + 8, _TILE_W, _TILE_H),
            "active": active == "Bob",
            "avatar_colour": (60, 60, 120),   # red-ish, H≈0 — outside green range
        },
    ]
    return make_webex_frame(participants, frame_w, frame_h)


def no_active_speaker_frame() -> np.ndarray:
    """Frame with two participants, neither active."""
    return two_participant_frame(active="")
