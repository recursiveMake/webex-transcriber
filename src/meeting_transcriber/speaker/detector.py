"""Per-frame speaker detection: green-mic detection + name OCR.

Design decisions:
- EasyOCR reader is created once and reused (expensive to initialise).
- Name results are cached by tile position to minimise OCR calls.
- MPS acceleration is used when available (Apple Silicon).
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .layout import (
    TileRegion,
    _build_green_mask,
    find_active_tiles,
    find_bordered_tiles,
    find_mic_blobs,
    infer_tile_from_mic,
    _blob_centre,
    _distance,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _get_reader():
    """Return an EasyOCR Reader, using MPS on Apple Silicon when available."""
    import easyocr
    try:
        import torch
        gpu = torch.backends.mps.is_available()
    except Exception:
        gpu = False
    device = "MPS" if gpu else "CPU"
    log.info("Initialising EasyOCR (device=%s) — this may take a moment…", device)
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    log.info("EasyOCR ready")
    return reader


# Module-level singleton with lock for thread-safe lazy initialisation.
_reader = None
_reader_lock = threading.Lock()


def get_reader():
    global _reader
    if _reader is None:
        with _reader_lock:
            # Double-checked locking: re-test inside the lock so two threads
            # that both see _reader is None don't both initialise it.
            if _reader is None:
                _reader = _get_reader()
    return _reader


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ActiveSpeaker:
    """A speaker detected as active in a single frame."""
    name: str
    tile: TileRegion
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Name cache
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    """A single name cache entry with hit tracking and soft-expiry state."""
    name: str
    confidence: float
    hits: int = 0
    soft_expired: bool = False


class _NameCache:
    """Caches OCR-derived participant names keyed by tile position.

    A position key is the rounded (mic_x / bucket, mic_y / bucket) cell, so
    small centroid jitter between frames maps to the same cache entry.

    Lifecycle
    ---------
    Active    → served from cache; hit counter increments each use.
    Soft-expired → ``get()`` returns None (triggering re-OCR) but the entry's
                   value is retained as a historical anchor.  Triggered by
                   either hitting ``reocr_interval`` or by a layout invalidation.
    Refreshed → re-OCR succeeded; entry becomes active again with hit=0.

    If re-OCR returns a *different* name for a soft-expired entry, that is a
    reliable signal that the layout has reflowed and a different person now
    occupies that tile position.

    If re-OCR consistently fails for a soft-expired entry, the historical
    value is used as a fallback (prevents an infinite OCR loop while the
    speaker is visible but OCR is struggling).
    """

    def __init__(self, bucket: int = 30, reocr_interval: int = 120) -> None:
        self._bucket = bucket
        self._reocr_interval = reocr_interval
        self._cache: dict[tuple[int, int], _CacheEntry] = {}

    def _key(self, tile: TileRegion) -> tuple[int, int]:
        # round() places bucket boundaries at midpoints rather than edges,
        # so jitter of a few pixels does not flip the key.
        return round(tile.mic_x / self._bucket), round(tile.mic_y / self._bucket)

    def get(self, tile: TileRegion) -> tuple[str, float] | None:
        """Return (name, confidence) if the entry is active; None otherwise.

        Returns None for missing entries and soft-expired entries alike.
        Use ``get_historical`` to retrieve the preserved value of a
        soft-expired entry (e.g. as a fallback when re-OCR fails).
        """
        key = self._key(tile)
        entry = self._cache.get(key)
        if entry is None:
            log.debug(
                "CACHE MISS  mic=(%d,%d) key=%s",
                tile.mic_x, tile.mic_y, key,
            )
            return None
        if entry.soft_expired:
            log.debug(
                "CACHE SOFT-EXPIRED  mic=(%d,%d) key=%s  name=%r — triggering re-OCR",
                tile.mic_x, tile.mic_y, key, entry.name,
            )
            return None
        if entry.hits >= self._reocr_interval:
            log.debug(
                "CACHE INTERVAL EXPIRED  mic=(%d,%d) key=%s  name=%r hits=%d — "
                "soft-expiring for re-OCR",
                tile.mic_x, tile.mic_y, key, entry.name, entry.hits,
            )
            entry.soft_expired = True
            return None
        entry.hits += 1
        log.debug(
            "CACHE HIT  mic=(%d,%d) key=%s  name=%r conf=%.2f hits=%d",
            tile.mic_x, tile.mic_y, key, entry.name, entry.confidence, entry.hits,
        )
        return entry.name, entry.confidence

    def get_historical(self, tile: TileRegion) -> tuple[str, float] | None:
        """Return the stored value even if soft-expired, for use as fallback."""
        key = self._key(tile)
        entry = self._cache.get(key)
        if entry is None:
            return None
        return entry.name, entry.confidence

    def put(self, tile: TileRegion, name: str, confidence: float) -> bool:
        """Store a name for this tile position.

        Returns True if the name differs from a soft-expired historical value
        at this position — a reliable signal of a post-reflow name change.
        """
        key = self._key(tile)
        entry = self._cache.get(key)
        name_changed = False

        if entry is not None and entry.soft_expired:
            if entry.name != name:
                log.info(
                    "CACHE RE-OCR MISMATCH  mic=(%d,%d) key=%s  "
                    "old=%r → new=%r — possible layout reflow",
                    tile.mic_x, tile.mic_y, key, entry.name, name,
                )
                name_changed = True
            else:
                log.debug(
                    "CACHE RE-OCR CONFIRMED  mic=(%d,%d) key=%s  name=%r conf=%.2f",
                    tile.mic_x, tile.mic_y, key, name, confidence,
                )
            # Refresh: replace with a new active entry.
            self._cache[key] = _CacheEntry(name=name, confidence=confidence)
        elif entry is None or confidence > entry.confidence:
            log.debug(
                "CACHE STORE  mic=(%d,%d) key=%s  name=%r conf=%.2f",
                tile.mic_x, tile.mic_y, key, name, confidence,
            )
            self._cache[key] = _CacheEntry(name=name, confidence=confidence)
        else:
            log.debug(
                "CACHE SKIP STORE  mic=(%d,%d) key=%s  name=%r "
                "new_conf=%.2f <= existing_conf=%.2f",
                tile.mic_x, tile.mic_y, key, name, confidence, entry.confidence,
            )

        return name_changed

    def soft_expire_all(self) -> None:
        """Mark every active entry as soft-expired (called on layout invalidation).

        Soft-expired entries retain their historical value for comparison and
        fallback; they are refreshed the next time re-OCR succeeds at their
        position.
        """
        count = sum(1 for e in self._cache.values() if not e.soft_expired)
        for entry in self._cache.values():
            entry.soft_expired = True
        log.info("CACHE SOFT-EXPIRED all %d active entries (layout invalidation)", count)

    def all_names(self) -> list[str]:
        return [e.name for e in self._cache.values()]


# ---------------------------------------------------------------------------
# OCR + name extraction
# ---------------------------------------------------------------------------

_NOISE_WORDS = frozenset({"mute", "unmute", "video", "more", "leave", "chat",
                           "participants", "share", "reactions", "webex", ""})


def _clean_name(raw: str) -> str:
    """Strip OCR noise and normalise a participant name string."""
    # Remove non-printable and purely numeric tokens
    tokens = re.split(r"\s+", raw.strip())
    tokens = [t for t in tokens if re.search(r"[a-zA-Z]", t)]
    tokens = [t for t in tokens if t.lower() not in _NOISE_WORDS]
    name = " ".join(tokens).strip()
    # Collapse multiple spaces
    name = re.sub(r"\s{2,}", " ", name)
    return name


def ocr_tile_name(
    frame: np.ndarray,
    tile: TileRegion,
    reader=None,
    *,
    min_confidence: float = 0.4,
) -> tuple[str, float]:
    """OCR the name region at the bottom of a participant tile.

    Returns (name_string, confidence).  Empty string if nothing readable.
    """
    if reader is None:
        reader = get_reader()

    h, w = frame.shape[:2]
    rx, ry, rw, rh = tile.name_roi(h, w)
    if rw <= 0 or rh <= 0:
        return "", 0.0

    roi = frame[ry: ry + rh, rx: rx + rw]
    if roi.size == 0:
        return "", 0.0

    # Upscale small ROIs to help OCR
    scale = max(1, 48 // rh)
    if scale > 1:
        roi = cv2.resize(roi, (rw * scale, rh * scale), interpolation=cv2.INTER_CUBIC)

    results = reader.readtext(roi, detail=1)
    if not results:
        return "", 0.0

    # Filter to confident detections and compute each region's x-centre + area.
    candidates: list[tuple[float, float, float, str]] = []  # (x_centre, area, conf, text)
    for bbox, text, conf in results:
        if conf < min_confidence:
            continue
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x_centre = float(np.mean(xs))
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        candidates.append((x_centre, area, conf, text))

    if not candidates:
        return "", 0.0

    # Pick the region with the largest area as the primary name.
    candidates.sort(key=lambda c: c[1], reverse=True)
    primary_x, primary_area, primary_conf, primary_text = candidates[0]
    primary_width = primary_area ** 0.5   # approximate

    # Include any other regions that are horizontally adjacent to the primary
    # (handles "First" and "Last" detected as separate regions by EasyOCR).
    # Exclude regions that are far away — they belong to a neighbouring tile.
    parts = [primary_text]
    conf_values = [primary_conf]
    for x_c, area, conf, text in candidates[1:]:
        if abs(x_c - primary_x) <= max(primary_width, 80):
            parts.append(text)
            conf_values.append(conf)

    # Sort parts left-to-right so "First Last" reads in the right order.
    ordered = sorted(zip([c[0] for c in candidates[:len(parts)]], parts), key=lambda t: t[0])
    raw = " ".join(p for _, p in ordered)
    name = _clean_name(raw)
    avg_conf = float(np.mean(conf_values))
    return name, avg_conf


# ---------------------------------------------------------------------------
# Per-frame speaker detection
# ---------------------------------------------------------------------------

class SpeakerDetector:
    """Detects active speakers in each frame using green mic + OCR.

    Args:
        cache_min_confidence: Names below this confidence are cached but
            preferentially replaced by higher-confidence reads later.
        mic_cluster_radius: Pixel radius within which two mic blobs are
            considered the same participant tile.
        on_layout_invalidate: Callback invoked when OCR fails for a
            soft-expired tile, signalling a stale layout.  Typically wired
            to LayoutTracker.invalidate so detection restarts from scratch.
    """

    def __init__(
        self,
        cache_min_confidence: float = 0.5,
        mic_cluster_radius: int = 40,
        reocr_interval: int = 120,
        on_layout_invalidate: "Callable[[float, str], None] | None" = None,
    ) -> None:
        self._cache = _NameCache(reocr_interval=reocr_interval)
        self._cache_min_confidence = cache_min_confidence
        self._mic_cluster_radius = mic_cluster_radius
        self._on_layout_invalidate = on_layout_invalidate

    def process_frame(
        self,
        frame: np.ndarray,
        layout_tiles: list[TileRegion] | None = None,
        timestamp: float = 0.0,
    ) -> list[ActiveSpeaker]:
        """Detect active speakers in a single BGR frame.

        Detection strategy (in priority order):

        1. **Border-first**: Find tiles by their green border rectangle.
           The border gives exact tile dimensions directly from the screen.
           Each bordered tile is matched to the nearest known layout tile
           (for cache-key stability) or treated as a new participant.

        2. **Multi-cue fallback**: If no bordered tiles are found, check all
           known layout tiles for any active-speaker cue (mic blob, border
           strip, name highlight).

        3. **Blob bootstrap**: If there are no known layout tiles at all, scan
           for mic blobs and infer tile positions as a cold-start mechanism.

        The green HSV mask is computed once and shared across all detection
        functions to avoid redundant colour-space conversions.

        Args:
            frame: BGR image array.
            layout_tiles: Accumulated known tile positions from LayoutTracker.
            timestamp: Frame timestamp, used for invalidation logging.

        Returns:
            List of ActiveSpeaker records (empty if no activity detected).
        """
        h, w = frame.shape[:2]

        # Compute the colour mask once; pass it to all detection functions.
        mask = _build_green_mask(frame)

        # --- Path 1: border-first detection ---
        bordered = find_bordered_tiles(frame, mask=mask)
        log.debug(
            "BORDER detection found %d tile(s): %s",
            len(bordered),
            [(t.mic_x, t.mic_y) for t in bordered],
        )

        active_tiles: list[TileRegion] = []

        if bordered:
            for bt in bordered:
                if layout_tiles:
                    closest = min(
                        layout_tiles,
                        key=lambda t: _distance(bt.centre, t.centre),
                    )
                    dist = _distance(bt.centre, closest.centre)
                    if dist <= self._mic_cluster_radius * 2:
                        log.debug(
                            "BORDER tile centre=(%d,%d) matched known tile mic=(%d,%d) "
                            "dist=%.1fpx — using stored tile for cache stability",
                            bt.centre[0], bt.centre[1],
                            closest.mic_x, closest.mic_y, dist,
                        )
                        if closest not in active_tiles:
                            active_tiles.append(closest)
                        continue
                log.debug(
                    "BORDER tile centre=(%d,%d) — new participant, using border bounds",
                    bt.centre[0], bt.centre[1],
                )
                active_tiles.append(bt)

        elif layout_tiles:
            # --- Path 2: multi-cue fallback on known layout ---
            log.debug(
                "No bordered tiles — multi-cue fallback on %d known tile(s)",
                len(layout_tiles),
            )
            active_tiles = find_active_tiles(frame, layout_tiles, mask=mask)
            log.debug(
                "FALLBACK multi-cue found %d active tile(s): %s",
                len(active_tiles),
                [(t.mic_x, t.mic_y) for t in active_tiles],
            )

        else:
            # --- Path 3: raw blob bootstrap (no layout yet) ---
            log.debug("No layout yet — raw blob detection only")
            blobs = find_mic_blobs(frame)
            if not blobs:
                return []
            blobs = _deduplicate_blobs(blobs, self._mic_cluster_radius)
            active_tiles = [infer_tile_from_mic(b, h, w) for b in blobs]

        if not active_tiles:
            return []

        reader = get_reader()
        speakers: list[ActiveSpeaker] = []

        for tile in active_tiles:
            cached = self._cache.get(tile)   # None if missing or soft-expired

            if cached and cached[1] >= self._cache_min_confidence:
                name, conf = cached
                log.debug(
                    "NAME RESOLVED from cache  mic=(%d,%d)  name=%r conf=%.2f",
                    tile.mic_x, tile.mic_y, name, conf,
                )
            else:
                log.debug(
                    "OCR REQUIRED  mic=(%d,%d)  cached=%s",
                    tile.mic_x, tile.mic_y,
                    f"{cached[0]!r} conf={cached[1]:.2f} (below threshold {self._cache_min_confidence})"
                    if cached else "none",
                )
                name, conf = ocr_tile_name(frame, tile, reader)
                if name:
                    log.info(
                        "OCR identified participant: %r conf=%.2f  mic=(%d,%d)",
                        name, conf, tile.mic_x, tile.mic_y,
                    )
                    self._cache.put(tile, name, conf)
                else:
                    # OCR failed — use historical value as fallback.
                    historical = self._cache.get_historical(tile)
                    if historical:
                        name, conf = historical
                        log.debug(
                            "OCR empty; using historical value %r  mic=(%d,%d)",
                            name, tile.mic_x, tile.mic_y,
                        )
                    else:
                        log.warning(
                            "Could not identify speaker at mic=(%d,%d) — "
                            "OCR empty and no cache entry",
                            tile.mic_x, tile.mic_y,
                        )
                        name, conf = "", 0.0

                    # Only trigger layout invalidation when OCR fails on a
                    # soft-expired tile (re-verify failed, position is stale).
                    # First-time OCR failures (cache miss) are normal and do
                    # not indicate a stale layout.
                    if self._on_layout_invalidate and self._cache.get_historical(tile):
                        log.info(
                            "OCR re-verify failed at mic=(%d,%d) — triggering layout invalidation",
                            tile.mic_x, tile.mic_y,
                        )
                        self._on_layout_invalidate(timestamp, "OCR re-verify failed on known tile")

            if name:
                speakers.append(ActiveSpeaker(name=name, tile=tile, confidence=conf))

        # Deduplicate by name: two tiles resolving to the same person (e.g. a
        # participant thumbnail and a name-bar overlay both reading as "Philip")
        # would otherwise cause repeated cache evictions that produce a cache
        # miss on every frame.  Keep the highest-confidence entry per name.
        if len(speakers) > 1:
            best: dict[str, ActiveSpeaker] = {}
            for s in speakers:
                if s.name not in best or s.confidence > best[s.name].confidence:
                    best[s.name] = s
            deduped = list(best.values())
            if len(deduped) < len(speakers):
                log.debug(
                    "Deduplicated %d→%d speakers: %s",
                    len(speakers), len(deduped),
                    [s.name for s in deduped],
                )
            speakers = deduped

        return speakers

    def invalidate_cache(self) -> None:
        """Soft-expire all cache entries (called when layout is invalidated).

        Entries are retained as historical anchors for comparison and fallback;
        they will be refreshed the next time OCR succeeds at their position.
        """
        self._cache.soft_expire_all()

    @property
    def known_participants(self) -> list[str]:
        """Return all participant names seen so far."""
        return self._cache.all_names()


def _deduplicate_blobs(
    blobs: list[tuple[int, int, int, int]],
    radius: int,
) -> list[tuple[int, int, int, int]]:
    """Merge blobs whose centres are within `radius` pixels, keeping largest."""
    if not blobs:
        return []
    blobs_sorted = sorted(blobs, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for blob in blobs_sorted:
        cx, cy = _blob_centre(blob)
        dominated = False
        for kept_blob in kept:
            kx, ky = _blob_centre(kept_blob)
            if _distance((cx, cy), (kx, ky)) <= radius:
                dominated = True
                break
        if not dominated:
            kept.append(blob)
    return kept
