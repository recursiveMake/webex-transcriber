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

def find_active_tiles(
    frame: np.ndarray,
    tiles: list[TileRegion],
) -> list[TileRegion]:
    """Return tiles where any WebEx active-speaker cue is detected.

    Checks three cues in order; any one is sufficient to flag a tile active:

    1. **Green mic-icon blob** within the tile (small, distinctive).
    2. **Green border/highlight** around the tile perimeter (thin outline
       drawn by WebEx around the active speaker's thumbnail).
    3. **Green name-strip highlight** at the bottom of the tile (the name
       label background turns green on the active speaker's tile in some
       WebEx versions).

    Computing the HSV green mask once and sharing it across all tiles keeps
    this O(frame_pixels + N_tiles) rather than O(N_tiles × frame_pixels).
    """
    if not tiles:
        return []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)
    fh, fw = frame.shape[:2]
    return [t for t in tiles if _tile_has_activity(green_mask, t, fh, fw)]


def _tile_has_activity(
    green_mask: np.ndarray,
    tile: TileRegion,
    fh: int,
    fw: int,
) -> bool:
    """Return True if any active-speaker cue fires for this tile's ROI."""
    x1, y1 = max(0, tile.x), max(0, tile.y)
    x2, y2 = min(fw, tile.x + tile.w), min(fh, tile.y + tile.h)
    if x2 <= x1 or y2 <= y1:
        return False

    roi = green_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    th, tw = y2 - y1, x2 - x1

    # --- Cue 1: mic-icon blob (small green region, most distinctive) ---
    # Morphological closing reconnects codec-fragmented pixels.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if _MIC_AREA_MIN <= cv2.contourArea(cnt) <= _MIC_AREA_MAX:
            return True

    # --- Cue 2: green border/outline around tile perimeter ---
    # WebEx draws a thin green rectangle around the active speaker's tile.
    # Use ~4% of each dimension as the border strip width.
    bw = max(3, tw // 25)
    bh = max(3, th // 25)
    top    = roi[:bh, :]
    bottom = roi[max(0, th - bh):, :]
    left   = roi[:, :bw]
    right  = roi[:, max(0, tw - bw):]
    perim_total = top.size + bottom.size + left.size + right.size
    if perim_total > 0:
        perim_green = (int(top.sum()) + int(bottom.sum()) +
                       int(left.sum()) + int(right.sum())) // 255
        if perim_green / perim_total > 0.08:   # ≥8% of perimeter is green
            return True

    # --- Cue 3: green-highlighted name strip at tile bottom ---
    # In some WebEx versions the entire bottom label bar turns green on the
    # active speaker's tile.  Check the bottom 30% of the tile ROI.
    name_top = int(th * 0.70)
    name_roi = roi[name_top:, :]
    if name_roi.size > 0:
        name_ratio = int(name_roi.sum()) // 255 / name_roi.size
        if name_ratio > 0.10:   # ≥10% of name strip is green
            return True

    return False


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

    # Accumulate mic-icon centres AND sizes across all frames.
    # Storing sizes lets us reconstruct a representative blob with realistic
    # dimensions rather than a hardcoded 10×10 placeholder.
    all_blobs: list[tuple[int, int, int, int]] = []  # (cx, cy, bw, bh)
    for _ts, frame in frames:
        for blob in find_mic_blobs(frame):
            bx, by, bw, bh = blob
            cx, cy = _blob_centre(blob)
            all_blobs.append((cx, cy, bw, bh))

    if not all_blobs:
        return None

    # Cluster nearby centres (simple greedy clustering)
    clusters: list[list[tuple[int, int, int, int]]] = []
    for item in all_blobs:
        cx, cy = item[0], item[1]
        placed = False
        for cluster in clusters:
            rep_cx, rep_cy = cluster[0][0], cluster[0][1]
            if _distance((cx, cy), (rep_cx, rep_cy)) <= cluster_radius:
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])

    # Keep only clusters that appeared in multiple frames
    stable = [c for c in clusters if len(c) >= min_blob_frames]
    if not stable:
        # Relax threshold for very short clips / sparse sampling
        stable = clusters

    # Representative centre + size per cluster (all medians)
    tiles: list[TileRegion] = []
    for cluster in stable:
        cx = int(np.median([item[0] for item in cluster]))
        cy = int(np.median([item[1] for item in cluster]))
        bw = int(np.median([item[2] for item in cluster]))
        bh = int(np.median([item[3] for item in cluster]))
        # Use actual observed blob dimensions for accurate tile inference
        half_w, half_h = max(1, bw // 2), max(1, bh // 2)
        blob = (cx - half_w, cy - half_h, bw, bh)
        tile = infer_tile_from_mic(blob, frame_h, frame_w)
        tiles.append(tile)

    ts = frames[len(frames) // 2][0]
    return LayoutSnapshot(timestamp=ts, tiles=tiles, frame_shape=(frame_h, frame_w))


# ---------------------------------------------------------------------------
# Layout tracker (handles drift over time)
# ---------------------------------------------------------------------------

class LayoutTracker:
    """Maintains and updates layout estimates as the video progresses.

    Tiles are **accumulated** over time: once a participant's tile position is
    discovered (because they spoke and showed a green cue), it is retained for
    the rest of the meeting.  New speakers are merged into the known set as they
    are detected; existing tile positions are updated if they drift significantly
    (screen share, participant panel reflow).

    This avoids the previous failure mode where the layout was replaced by a
    smaller snapshot — e.g. if only 1 person spoke during the first window, only
    1 tile would be recorded and all other participants would be invisible.
    """

    def __init__(
        self,
        window: int = 10,
        drift_threshold: int = 80,
    ) -> None:
        self._window = window
        self._drift_threshold = drift_threshold
        self._buffer: list[tuple[float, np.ndarray]] = []
        self._tiles: list[TileRegion] = []   # accumulated across entire meeting
        self._frame_shape: tuple[int, int] = (0, 0)

    @property
    def current(self) -> LayoutSnapshot | None:
        if not self._tiles:
            return None
        return LayoutSnapshot(
            timestamp=0.0,
            tiles=list(self._tiles),
            frame_shape=self._frame_shape,
        )

    def update(self, timestamp: float, frame: np.ndarray) -> LayoutSnapshot | None:
        """Feed a new frame; returns the accumulated LayoutSnapshot."""
        self._buffer.append((timestamp, frame))
        if len(self._buffer) > self._window:
            self._buffer.pop(0)

        if len(self._buffer) < max(2, self._window // 3):
            return self.current  # not enough data yet

        if self._frame_shape == (0, 0):
            self._frame_shape = frame.shape[:2]

        candidate = detect_layout(self._buffer)
        if candidate is None:
            return self.current

        self._merge(candidate.tiles, timestamp)
        return self.current

    def add_tile(self, tile: TileRegion, timestamp: float) -> None:
        """Immediately merge a single confirmed tile into the accumulated layout.

        Called by the pipeline when OCR successfully identifies a speaker found
        via the secondary blob-scan path.  Once the tile is in the layout, future
        frames will use the primary ``find_active_tiles`` path with a stable,
        stored position rather than re-inferring a fresh tile each frame.
        """
        self._merge([tile], timestamp)

    def _merge(self, new_tiles: list[TileRegion], timestamp: float) -> None:
        """Merge newly detected tiles into the accumulated set.

        For each new tile:
        - If it is close to an existing tile (within drift_threshold), update
          the existing tile's position if it has moved significantly.
        - If it is far from all existing tiles, add it as a new participant.
        """
        for new_tile in new_tiles:
            closest: TileRegion | None = None
            closest_dist = float("inf")
            for existing in self._tiles:
                d = _distance(existing.centre, new_tile.centre)
                if d < closest_dist:
                    closest_dist = d
                    closest = existing

            if closest is None or closest_dist > self._drift_threshold:
                # Never-seen position — new participant tile
                log.info(
                    "New participant tile at %.1fs: mic position (%d, %d)",
                    timestamp, new_tile.mic_x, new_tile.mic_y,
                )
                self._tiles.append(new_tile)
            elif closest_dist > self._drift_threshold / 2:
                # Same participant, position has drifted — update in place
                log.info(
                    "Tile position updated at %.1fs: moved %.0fpx",
                    timestamp, closest_dist,
                )
                idx = self._tiles.index(closest)
                self._tiles[idx] = new_tile
