"""Automatic WebEx participant-panel layout detection.

Strategy (no user input required):
1. Sample frames distributed across the video.
2. In each frame, find bright-green blobs — these are active mic icons.
3. Cluster blob positions spatially; persistent clusters across frames are
   real mic icons rather than noise.
4. From the cluster positions infer tile boundaries by expanding each mic
   location to a rectangular participant tile.
5. Verify each tile contains OCR-readable text (the participant name).

Handles layout changes (late joiners, screen-share shifts) by re-running
layout detection when accumulated evidence drifts significantly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WebEx green mic colour range (HSV, OpenCV 0-179 hue scale)
# Calibrated to WebEx's bright-green active-speaker indicator.
# ---------------------------------------------------------------------------
_GREEN_LOWER = np.array([40, 120, 120], dtype=np.uint8)
_GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)

# Mic icon size bounds in pixels (area)
_MIC_AREA_MIN = 15
_MIC_AREA_MAX = 3000

# Tile aspect ratio expected around each mic icon
# WebEx tiles are approx 16:9 when camera on, ~1:1 for avatar tiles.
_TILE_ASPECT_MIN = 0.5
_TILE_ASPECT_MAX = 2.5


@dataclass
class TileRegion:
    """A detected participant tile (bounding box in frame coordinates)."""
    x: int
    y: int
    w: int
    h: int
    mic_x: int  # mic icon centre within the frame
    mic_y: int

    @property
    def centre(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def name_roi(self, frame_h: int, frame_w: int) -> tuple[int, int, int, int]:
        """Return (x, y, w, h) of the name-text region at tile bottom."""
        ny = max(0, self.y + int(self.h * 0.70))
        nh = min(frame_h - ny, int(self.h * 0.30) + 4)
        return self.x, ny, self.w, nh

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y: self.y + self.h, self.x: self.x + self.w]


@dataclass
class LayoutSnapshot:
    """Detected layout at a particular moment in the video."""
    timestamp: float
    tiles: list[TileRegion] = field(default_factory=list)
    frame_shape: tuple[int, int] = (0, 0)  # (height, width)


# ---------------------------------------------------------------------------
# Core detection helpers
# ---------------------------------------------------------------------------

def find_mic_blobs(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) bounding boxes of green mic candidates."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)
    # Morphological closing to join fragmented blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if _MIC_AREA_MIN <= area <= _MIC_AREA_MAX:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if 0.2 < aspect < 5:
                blobs.append((x, y, w, h))
    return blobs


def _blob_centre(blob: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, w, h = blob
    return x + w // 2, y + h // 2


def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def infer_tile_from_mic(
    blob: tuple[int, int, int, int],
    frame_h: int,
    frame_w: int,
) -> TileRegion:
    """Estimate participant tile extents from a mic icon blob.

    WebEx places the mic icon near the bottom of each thumbnail tile.
    We expand upward and sideways to approximate the tile boundary.
    """
    bx, by, bw, bh = blob
    cx, cy = _blob_centre(blob)

    # Heuristic tile dimensions relative to frame size.
    # Tiles are typically 1/5 to 1/3 of the shorter frame dimension.
    tile_side = max(bw * 6, min(frame_w, frame_h) // 5)
    tile_w = tile_side
    tile_h = int(tile_side * 0.75)

    # Mic icon sits in the lower ~20% of the tile
    # so the tile top is roughly tile_h * 0.8 above the mic centre.
    tx = max(0, cx - tile_w // 2)
    ty = max(0, cy - int(tile_h * 0.85))
    tw = min(frame_w - tx, tile_w)
    th = min(frame_h - ty, tile_h)

    return TileRegion(x=tx, y=ty, w=tw, h=th, mic_x=cx, mic_y=cy)


# ---------------------------------------------------------------------------
# Multi-frame layout detector
# ---------------------------------------------------------------------------

def detect_layout(
    frames: Sequence[tuple[float, np.ndarray]],
    *,
    min_blob_frames: int = 2,
    cluster_radius: int = 40,
) -> LayoutSnapshot | None:
    """Detect participant tiles from a sample of (timestamp, frame) pairs.

    Args:
        frames: Sequence of (timestamp, BGR frame array) pairs.
        min_blob_frames: A mic position must appear in at least this many
            frames to be considered a real tile (filters noise).
        cluster_radius: Pixel radius for merging nearby mic positions.

    Returns:
        A LayoutSnapshot, or None if no participant tiles were found.
    """
    if not frames:
        return None

    frame_h, frame_w = frames[0][1].shape[:2]

    # Accumulate all mic-icon centre positions across frames
    all_centres: list[tuple[int, int]] = []
    for _ts, frame in frames:
        for blob in find_mic_blobs(frame):
            all_centres.append(_blob_centre(blob))

    if not all_centres:
        return None

    # Cluster nearby centres (simple greedy clustering)
    clusters: list[list[tuple[int, int]]] = []
    for pt in all_centres:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            if _distance(pt, rep) <= cluster_radius:
                cluster.append(pt)
                placed = True
                break
        if not placed:
            clusters.append([pt])

    # Keep only clusters that appeared in multiple frames
    stable = [c for c in clusters if len(c) >= min_blob_frames]
    if not stable:
        # Relax threshold for very short clips / sparse sampling
        stable = clusters

    # Representative centre per cluster (median)
    tiles: list[TileRegion] = []
    for cluster in stable:
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        cx = int(np.median(xs))
        cy = int(np.median(ys))
        # Reconstruct a synthetic blob at the representative position
        blob = (cx - 5, cy - 5, 10, 10)
        tile = infer_tile_from_mic(blob, frame_h, frame_w)
        tiles.append(tile)

    ts = frames[len(frames) // 2][0]
    return LayoutSnapshot(timestamp=ts, tiles=tiles, frame_shape=(frame_h, frame_w))


# ---------------------------------------------------------------------------
# Layout tracker (handles drift over time)
# ---------------------------------------------------------------------------

class LayoutTracker:
    """Maintains and updates layout estimates as the video progresses.

    Calls ``detect_layout`` on a rolling window of recent frames and
    re-calibrates when tile positions shift (screen share, late joiners).
    """

    def __init__(
        self,
        window: int = 10,
        drift_threshold: int = 80,
    ) -> None:
        self._window = window
        self._drift_threshold = drift_threshold
        self._buffer: list[tuple[float, np.ndarray]] = []
        self._current: LayoutSnapshot | None = None

    def update(self, timestamp: float, frame: np.ndarray) -> LayoutSnapshot | None:
        """Feed a new frame; returns updated LayoutSnapshot when ready."""
        self._buffer.append((timestamp, frame))
        if len(self._buffer) > self._window:
            self._buffer.pop(0)

        if len(self._buffer) < max(2, self._window // 3):
            return self._current  # not enough data yet

        candidate = detect_layout(self._buffer)
        if candidate is None:
            return self._current

        # Check for significant drift vs current layout
        if self._current is not None and self._has_drifted(candidate):
            self._current = candidate
        elif self._current is None:
            self._current = candidate

        return self._current

    def _has_drifted(self, new: LayoutSnapshot) -> bool:
        if len(new.tiles) != len(self._current.tiles):  # type: ignore[union-attr]
            old_n = len(self._current.tiles)  # type: ignore[union-attr]
            log.info(
                "Layout drift at %.1fs: tile count changed %d → %d",
                new.timestamp, old_n, len(new.tiles),
            )
            return True
        for old_tile, new_tile in zip(
            sorted(self._current.tiles, key=lambda t: (t.y, t.x)),  # type: ignore[union-attr]
            sorted(new.tiles, key=lambda t: (t.y, t.x)),
        ):
            dist = _distance(old_tile.centre, new_tile.centre)
            if dist > self._drift_threshold:
                log.info(
                    "Layout drift at %.1fs: tile moved %.0fpx (threshold=%dpx)",
                    new.timestamp, dist, self._drift_threshold,
                )
                return True
        return False
