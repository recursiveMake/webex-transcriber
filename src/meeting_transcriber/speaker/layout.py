"""Automatic WebEx participant-panel layout detection.

Strategy (no user input required):
1. Sample frames distributed across the video.
2. In each frame, find green-bordered tile rectangles — these are active
   participant tiles as drawn by WebEx.
3. Cluster tile positions across frames; persistent positions are real tiles.
4. Accumulate discovered tiles over time; invalidate on layout drift.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WebEx green / teal active-speaker colour range (HSV, OpenCV 0-179 hue scale)
# Widened to H:40-110 to capture the teal-leaning border variant seen in some
# WebEx versions; S/V floors lowered to 100 to catch slightly desaturated cues.
# ---------------------------------------------------------------------------
_GREEN_LOWER = np.array([40, 100, 100], dtype=np.uint8)
_GREEN_UPPER = np.array([110, 255, 255], dtype=np.uint8)

# Mic icon size bounds in pixels (area and linear dimension).
_MIC_AREA_MIN = 15
_MIC_AREA_MAX = 1200   # large blobs are borders or noise, not mic icons
_MIC_DIM_MIN = 4       # minimum width or height in pixels
_MIC_DIM_MAX = 40      # maximum width or height in pixels

# Minimum border rectangle dimensions — smaller candidates are noise.
_BORDER_MIN_W = 40   # pixels
_BORDER_MIN_H = 30   # pixels

# A hollow border has most of its bounding-rect area NOT filled with green.
# Filled blobs (mic icon, name strip) have a higher fill ratio.
_BORDER_HOLLOW_RATIO = 0.5


@dataclass
class TileRegion:
    """A detected participant tile (bounding box in frame coordinates)."""
    x: int
    y: int
    w: int
    h: int
    mic_x: int  # estimated mic icon centre within the frame
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
# Shared colour mask
# ---------------------------------------------------------------------------

def _build_green_mask(frame: np.ndarray) -> np.ndarray:
    """Return the binary green/teal HSV mask for a BGR frame.

    Called once per frame and shared across all detection functions so the
    expensive colour-space conversion is not repeated.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)


# ---------------------------------------------------------------------------
# Border-first tile detection (primary)
# ---------------------------------------------------------------------------

def find_bordered_tiles(
    frame: np.ndarray,
    mask: np.ndarray | None = None,
) -> list[TileRegion]:
    """Detect participant tiles by their green border rectangles.

    WebEx draws a green (or teal) rectangular outline around the active
    speaker's tile.  Finding that outline directly gives the tile's true pixel
    boundaries — far more reliable than inferring size from a mic blob.

    Args:
        frame: BGR frame array.
        mask: Pre-computed green HSV mask (computed if not provided).

    Returns:
        List of TileRegion, one per detected bordered tile.  Tiles covering
        more than 50% of the frame area are rejected as artifacts.
    """
    fh, fw = frame.shape[:2]
    max_tile_area = fw * fh * 0.5

    if mask is None:
        mask = _build_green_mask(frame)

    # Dilate slightly to connect fragmented border segments.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tiles: list[TileRegion] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < _BORDER_MIN_W or h < _BORDER_MIN_H:
            continue
        if w * h > max_tile_area:
            log.debug(
                "find_bordered_tiles: rejected oversized candidate %dx%d at (%d,%d)",
                w, h, x, y,
            )
            continue

        # A border is hollow — most of its bounding rect is NOT green.
        # Use actual pixel count (not contourArea, which measures enclosed area
        # and is ~w*h for any convex shape regardless of fill).
        filled_pixels = cv2.countNonZero(dilated[y: y + h, x: x + w])
        if w * h > 0 and (filled_pixels / (w * h)) > _BORDER_HOLLOW_RATIO:
            continue  # filled blob, not a border

        # Mic icon sits near the bottom centre of the tile.
        mic_x = x + w // 2
        mic_y = y + h - max(4, h // 10)

        tiles.append(TileRegion(x=x, y=y, w=w, h=h, mic_x=mic_x, mic_y=mic_y))
        log.debug(
            "find_bordered_tiles: tile %dx%d at (%d,%d) mic=(%d,%d)",
            w, h, x, y, mic_x, mic_y,
        )

    return tiles


# ---------------------------------------------------------------------------
# Multi-cue activity check on known tiles (fallback)
# ---------------------------------------------------------------------------

def find_active_tiles(
    frame: np.ndarray,
    tiles: list[TileRegion],
    mask: np.ndarray | None = None,
) -> list[TileRegion]:
    """Return tiles where any WebEx active-speaker cue is detected.

    Checks three cues (mic blob, perimeter border, name-strip highlight).
    Used as a fallback when border-first detection finds nothing.

    Args:
        frame: BGR frame array.
        tiles: Known tile positions to check.
        mask: Pre-computed green HSV mask (computed if not provided).
    """
    if not tiles:
        return []
    if mask is None:
        mask = _build_green_mask(frame)
    fh, fw = frame.shape[:2]
    return [t for t in tiles if _tile_has_activity(mask, t, fh, fw)]


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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if _MIC_AREA_MIN <= cv2.contourArea(cnt) <= _MIC_AREA_MAX:
            return True

    # --- Cue 2: green border around tile perimeter ---
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
        if perim_green / perim_total > 0.08:
            return True

    # --- Cue 3: green-highlighted name strip at tile bottom ---
    name_top = int(th * 0.70)
    name_roi = roi[name_top:, :]
    if name_roi.size > 0:
        name_ratio = int(name_roi.sum()) // 255 / name_roi.size
        if name_ratio > 0.10:
            return True

    return False


# ---------------------------------------------------------------------------
# Mic blob helpers (bootstrap path only)
# ---------------------------------------------------------------------------

def find_mic_blobs(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) bounding boxes of green mic-icon candidates.

    Used only as a last-resort bootstrap when no layout is known and no
    bordered tiles are detected.
    """
    mask = _build_green_mask(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # RETR_LIST (not RETR_EXTERNAL) so mic blobs nested inside a border ring
    # are still returned as individual contours.
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if _MIC_AREA_MIN <= area <= _MIC_AREA_MAX:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < _MIC_DIM_MIN or h < _MIC_DIM_MIN:
                continue
            if w > _MIC_DIM_MAX or h > _MIC_DIM_MAX:
                continue
            if 0.2 < (w / h if h > 0 else 0) < 5:
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

    Tile dimensions are derived from frame size only — never from blob
    dimensions, which can be unreliable.  Used only as a bootstrap fallback
    when neither bordered tiles nor a known layout are available.
    """
    cx, cy = _blob_centre(blob)

    tile_side = min(frame_w, frame_h) // 5
    tile_w = tile_side
    tile_h = int(tile_side * 0.75)

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
    min_tile_frames: int = 2,
    cluster_radius: int = 40,
) -> LayoutSnapshot | None:
    """Detect participant tiles from a sample of (timestamp, frame) pairs.

    Uses border detection on each frame — consistent with the primary per-frame
    detection path.  Tiles that appear in multiple frames are considered stable.

    Args:
        frames: Sequence of (timestamp, BGR frame array) pairs.
        min_tile_frames: A tile position must appear in at least this many
            frames to be considered stable (filters single-frame noise).
        cluster_radius: Pixel radius for merging nearby tile centres.

    Returns:
        A LayoutSnapshot, or None if no tiles were found.
    """
    if not frames:
        return None

    frame_h, frame_w = frames[0][1].shape[:2]

    # Collect all bordered tiles across the buffer.
    all_tiles: list[TileRegion] = []
    for _ts, frame in frames:
        all_tiles.extend(find_bordered_tiles(frame))

    if not all_tiles:
        return None

    # Cluster tiles by centre position.
    clusters: list[list[TileRegion]] = []
    for tile in all_tiles:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            if _distance(tile.centre, rep.centre) <= cluster_radius:
                cluster.append(tile)
                placed = True
                break
        if not placed:
            clusters.append([tile])

    stable = [c for c in clusters if len(c) >= min_tile_frames]
    if not stable:
        stable = clusters  # relax for short clips / sparse sampling

    # Representative tile per cluster: median of all observed bounds.
    tiles: list[TileRegion] = []
    for cluster in stable:
        tiles.append(TileRegion(
            x=int(np.median([t.x for t in cluster])),
            y=int(np.median([t.y for t in cluster])),
            w=int(np.median([t.w for t in cluster])),
            h=int(np.median([t.h for t in cluster])),
            mic_x=int(np.median([t.mic_x for t in cluster])),
            mic_y=int(np.median([t.mic_y for t in cluster])),
        ))

    ts = frames[len(frames) // 2][0]
    return LayoutSnapshot(timestamp=ts, tiles=tiles, frame_shape=(frame_h, frame_w))


# ---------------------------------------------------------------------------
# Layout tracker (handles drift over time)
# ---------------------------------------------------------------------------

class LayoutTracker:
    """Maintains and updates layout estimates as the video progresses.

    The ``on_invalidate`` callback is called whenever the tile set is cleared
    so the name cache can be soft-expired in sync with the layout reset.

    Between resets, tiles are accumulated: new speakers are merged in as they
    are discovered, and positions that drift are updated in place.
    """

    def __init__(
        self,
        window: int = 10,
        drift_threshold: int = 80,
        on_invalidate: "Callable[[], None] | None" = None,
    ) -> None:
        self._window = window
        self._drift_threshold = drift_threshold
        self._on_invalidate = on_invalidate
        self._buffer: list[tuple[float, np.ndarray]] = []
        self._tiles: list[TileRegion] = []
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

    def invalidate(self, timestamp: float, reason: str) -> None:
        """Clear all accumulated tiles and notify the cache to soft-expire."""
        log.info(
            "Layout invalidated at %.1fs (%s) — clearing %d tile(s) and resetting",
            timestamp, reason, len(self._tiles),
        )
        self._tiles.clear()
        self._buffer.clear()
        if self._on_invalidate:
            self._on_invalidate()

    def update(self, timestamp: float, frame: np.ndarray) -> LayoutSnapshot | None:
        """Feed a new frame; returns the current LayoutSnapshot."""
        self._buffer.append((timestamp, frame))
        if len(self._buffer) > self._window:
            self._buffer.pop(0)

        if self._frame_shape == (0, 0):
            self._frame_shape = frame.shape[:2]

        if len(self._buffer) < max(2, self._window // 3):
            return self.current

        candidate = detect_layout(self._buffer)
        if candidate is None:
            return self.current

        self._merge(candidate.tiles, timestamp)
        return self.current

    def add_tile(self, tile: TileRegion, timestamp: float) -> None:
        """Immediately merge a single confirmed tile into the accumulated layout."""
        self._merge([tile], timestamp)

    def _merge(self, new_tiles: list[TileRegion], timestamp: float) -> None:
        """Merge newly detected tiles into the accumulated set."""
        for new_tile in new_tiles:
            closest: TileRegion | None = None
            closest_dist = float("inf")
            for existing in self._tiles:
                d = _distance(existing.centre, new_tile.centre)
                if d < closest_dist:
                    closest_dist = d
                    closest = existing

            if closest is None or closest_dist > self._drift_threshold:
                log.info(
                    "New participant tile at %.1fs: mic position (%d, %d)",
                    timestamp, new_tile.mic_x, new_tile.mic_y,
                )
                self._tiles.append(new_tile)
            elif closest_dist > self._drift_threshold / 2:
                log.info(
                    "Tile position updated at %.1fs: moved %.0fpx",
                    timestamp, closest_dist,
                )
                idx = self._tiles.index(closest)
                self._tiles[idx] = new_tile
