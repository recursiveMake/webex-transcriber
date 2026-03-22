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
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .layout import (
    TileRegion,
    find_active_tiles,
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

class _NameCache:
    """Caches OCR-derived participant names keyed by tile position.

    A position key is the (mic_x // bucket, mic_y // bucket) cell, so
    small jitter between frames maps to the same cache entry.

    Each entry tracks how many times it has been served from cache without
    re-OCR verification.  After ``reocr_interval`` hits the entry is treated
    as expired: the next detection at that position runs OCR again.  This
    catches the case where a layout reflow places a *different* person at a
    position previously associated with someone else — even when the pixel
    coordinates did not change (so the LayoutTracker cannot detect a shift).
    """

    def __init__(self, bucket: int = 30, reocr_interval: int = 120) -> None:
        self._bucket = bucket
        self._reocr_interval = reocr_interval
        # Value: (name, confidence, hit_count)
        self._cache: dict[tuple[int, int], tuple[str, float, int]] = {}

    def _key(self, tile: TileRegion) -> tuple[int, int]:
        # round() rather than // places the bucket boundary at the midpoint
        # (bucket/2 px) instead of the edge (0 px), so blob centroid jitter
        # of a few pixels no longer flips the key at a boundary.
        return round(tile.mic_x / self._bucket), round(tile.mic_y / self._bucket)

    def get(self, tile: TileRegion) -> tuple[str, float] | None:
        """Return cached (name, confidence), or None if missing or expired."""
        key = self._key(tile)
        entry = self._cache.get(key)
        if entry is None:
            log.debug(
                "CACHE MISS  mic=(%d,%d) key=%s  cache_keys=%s",
                tile.mic_x, tile.mic_y, key,
                list(self._cache.keys()),
            )
            return None
        name, conf, hits = entry
        if hits >= self._reocr_interval:
            log.debug(
                "CACHE EXPIRED  mic=(%d,%d) key=%s  name=%r hits=%d",
                tile.mic_x, tile.mic_y, key, name, hits,
            )
            return None
        self._cache[key] = (name, conf, hits + 1)
        log.debug(
            "CACHE HIT  mic=(%d,%d) key=%s  name=%r conf=%.2f hits=%d",
            tile.mic_x, tile.mic_y, key, name, conf, hits + 1,
        )
        return name, conf

    def put(self, tile: TileRegion, name: str, confidence: float) -> None:
        key = self._key(tile)
        # If this name is already known at a different position (layout reflow
        # shifted the participant's tile), remove the stale entry so we don't
        # accumulate duplicate keys for the same person.
        for stale_key, (stale_name, _, _) in list(self._cache.items()):
            if stale_name == name and stale_key != key:
                log.debug(
                    "CACHE EVICT stale key=%s for %r (new key=%s mic=(%d,%d))",
                    stale_key, name, key, tile.mic_x, tile.mic_y,
                )
                del self._cache[stale_key]
                break
        existing = self._cache.get(key)
        if existing is None or confidence > existing[1]:
            log.debug(
                "CACHE STORE  mic=(%d,%d) key=%s  name=%r conf=%.2f",
                tile.mic_x, tile.mic_y, key, name, confidence,
            )
            self._cache[key] = (name, confidence, 0)  # reset hit counter
        else:
            log.debug(
                "CACHE SKIP STORE  mic=(%d,%d) key=%s  name=%r "
                "new_conf=%.2f <= existing_conf=%.2f",
                tile.mic_x, tile.mic_y, key, name, confidence, existing[1],
            )

    def all_names(self) -> list[str]:
        return [name for name, _, _ in self._cache.values()]


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
    """

    def __init__(
        self,
        cache_min_confidence: float = 0.5,
        mic_cluster_radius: int = 40,
        reocr_interval: int = 120,
    ) -> None:
        self._cache = _NameCache(reocr_interval=reocr_interval)
        self._cache_min_confidence = cache_min_confidence
        self._mic_cluster_radius = mic_cluster_radius

    def process_frame(
        self,
        frame: np.ndarray,
        layout_tiles: list[TileRegion] | None = None,
        allow_new_ocr: bool = True,
    ) -> list[ActiveSpeaker]:
        """Detect active speakers in a single BGR frame.

        Args:
            frame: BGR image array.
            layout_tiles: Pre-detected tile regions (optional).  When provided,
                all three WebEx cues (mic icon, tile border, name highlight) are
                checked for each known tile.  When None, falls back to raw
                green-blob detection across the whole frame.
            allow_new_ocr: When False, skip OCR for unseen tile positions and
                rely solely on the name cache.  Used outside the 5–95% video
                window to avoid OCR calls during dynamic meeting start/end.
                Tiles with no cached name are silently dropped in this mode.

        Returns:
            List of ActiveSpeaker records (empty if no activity detected).
        """
        h, w = frame.shape[:2]

        if layout_tiles:
            log.debug(
                "LAYOUT known tiles: %s",
                [(t.mic_x, t.mic_y) for t in layout_tiles],
            )

            # Primary path: check all three cues on known tile positions.
            active_tiles = find_active_tiles(frame, layout_tiles)
            log.debug(
                "PRIMARY path found %d active tile(s): %s",
                len(active_tiles),
                [(t.mic_x, t.mic_y) for t in active_tiles],
            )

            # Secondary path: also scan the full frame for blobs to catch
            # speakers whose tile may not have fired via the multi-cue check,
            # or who are genuinely new and not yet in the layout.
            #
            # Use mic-position proximity rather than tile-boundary containment.
            # Tile boundaries are heuristic and the blob centroid can sit just
            # outside them even for a known participant — causing a fresh tile
            # to be inferred each frame with a slightly different mic_x/mic_y,
            # producing a different cache key and triggering OCR every frame.
            # Matching by proximity to the stored mic_x/mic_y gives a stable
            # reference and therefore a stable cache key.
            blobs = find_mic_blobs(frame)
            blobs = _deduplicate_blobs(blobs, self._mic_cluster_radius)
            log.debug(
                "SECONDARY blob scan found %d blob(s): %s",
                len(blobs),
                [_blob_centre(b) for b in blobs],
            )
            for blob in blobs:
                cx, cy = _blob_centre(blob)
                closest = min(
                    layout_tiles,
                    key=lambda t: _distance((cx, cy), (t.mic_x, t.mic_y)),
                    default=None,
                )
                if closest is not None:
                    dist = _distance((cx, cy), (closest.mic_x, closest.mic_y))
                    log.debug(
                        "SECONDARY blob (%d,%d) → closest known mic=(%d,%d) dist=%.1fpx "
                        "(radius=%dpx)",
                        cx, cy, closest.mic_x, closest.mic_y, dist,
                        self._mic_cluster_radius,
                    )
                    if dist <= self._mic_cluster_radius:
                        if closest not in active_tiles:
                            log.debug(
                                "SECONDARY blob (%d,%d) matched known tile mic=(%d,%d) "
                                "— using stored tile",
                                cx, cy, closest.mic_x, closest.mic_y,
                            )
                            active_tiles.append(closest)
                        else:
                            log.debug(
                                "SECONDARY blob (%d,%d) matched known tile mic=(%d,%d) "
                                "— already in active_tiles",
                                cx, cy, closest.mic_x, closest.mic_y,
                            )
                    else:
                        log.debug(
                            "SECONDARY blob (%d,%d) dist=%.1fpx > radius=%dpx "
                            "— inferring new tile",
                            cx, cy, dist, self._mic_cluster_radius,
                        )
                        active_tiles.append(infer_tile_from_mic(blob, h, w))
                else:
                    log.debug(
                        "SECONDARY blob (%d,%d) — no known tiles, inferring new tile",
                        cx, cy,
                    )
                    active_tiles.append(infer_tile_from_mic(blob, h, w))
        else:
            log.debug("No layout yet — raw blob detection only")
            # No layout knowledge yet — raw blob detection only.
            blobs = find_mic_blobs(frame)
            if not blobs:
                return []
            blobs = _deduplicate_blobs(blobs, self._mic_cluster_radius)
            active_tiles = [infer_tile_from_mic(blob, h, w) for blob in blobs]

        if not active_tiles:
            return []

        reader = get_reader() if allow_new_ocr else None
        speakers: list[ActiveSpeaker] = []

        for tile in active_tiles:
            cached = self._cache.get(tile)

            if cached and cached[1] >= self._cache_min_confidence:
                name, conf = cached
                log.debug(
                    "NAME RESOLVED from cache  mic=(%d,%d)  name=%r conf=%.2f",
                    tile.mic_x, tile.mic_y, name, conf,
                )
            elif allow_new_ocr:
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
                elif cached:
                    name, conf = cached
                    log.debug(
                        "OCR returned empty; falling back to cached %r  mic=(%d,%d)",
                        name, tile.mic_x, tile.mic_y,
                    )
                else:
                    log.warning(
                        "Could not identify speaker at mic=(%d,%d) — "
                        "OCR empty and no cache entry",
                        tile.mic_x, tile.mic_y,
                    )
                    name, conf = "", 0.0
            else:
                if cached:
                    name, conf = cached
                else:
                    name, conf = "", 0.0

            if name:
                speakers.append(ActiveSpeaker(name=name, tile=tile, confidence=conf))

        return speakers

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
