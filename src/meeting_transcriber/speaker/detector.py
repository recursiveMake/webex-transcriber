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
    """

    def __init__(self, bucket: int = 30) -> None:
        self._bucket = bucket
        self._cache: dict[tuple[int, int], tuple[str, float]] = {}

    def _key(self, tile: TileRegion) -> tuple[int, int]:
        return tile.mic_x // self._bucket, tile.mic_y // self._bucket

    def get(self, tile: TileRegion) -> tuple[str, float] | None:
        return self._cache.get(self._key(tile))

    def put(self, tile: TileRegion, name: str, confidence: float) -> None:
        key = self._key(tile)
        existing = self._cache.get(key)
        if existing is None or confidence > existing[1]:
            self._cache[key] = (name, confidence)

    def all_names(self) -> list[str]:
        return [name for name, _ in self._cache.values()]


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

    texts: list[str] = []
    confidences: list[float] = []
    for _bbox, text, conf in results:
        if conf >= min_confidence:
            texts.append(text)
            confidences.append(conf)

    if not texts:
        return "", 0.0

    raw = " ".join(texts)
    name = _clean_name(raw)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
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
    ) -> None:
        self._cache = _NameCache()
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
            # Multi-cue detection: mic icon OR tile border OR name highlight.
            # Tiles come directly from the layout snapshot (stable mic_x/mic_y
            # used as the name-cache key).
            active_tiles = find_active_tiles(frame, layout_tiles)
        else:
            # No layout knowledge yet — fall back to raw blob detection.
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
                    "Cache hit for tile (%d,%d): %r (conf=%.2f)",
                    tile.mic_x, tile.mic_y, name, conf,
                )
            elif allow_new_ocr:
                name, conf = ocr_tile_name(frame, tile, reader)
                if name:
                    log.info("OCR identified new participant: %r (conf=%.2f)", name, conf)
                    self._cache.put(tile, name, conf)
                elif cached:
                    name, conf = cached
                    log.debug(
                        "OCR failed; using cached name %r for tile (%d,%d)",
                        name, tile.mic_x, tile.mic_y,
                    )
                else:
                    log.warning(
                        "Could not identify speaker at tile (%d,%d) — "
                        "no OCR result and no cache",
                        tile.mic_x, tile.mic_y,
                    )
                    name, conf = "", 0.0
            else:
                # OCR disabled for this frame (outside the 5–95% window).
                # Use lower-confidence cached name if available; otherwise drop.
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
