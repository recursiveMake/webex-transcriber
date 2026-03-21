"""Build a speaker activity timeline from per-frame detection results.

The timeline maps time intervals to lists of active speaker names.
Consecutive frames with the same speaker(s) are merged into spans.

Timestamp resilience
--------------------
WebEx has a known visual lag: the green mic indicator appears ~200-400ms
after a speaker starts and clears ~100-300ms after they stop.  Frame
sampling at 1 fps adds additional quantisation error (up to ±frame_interval).
We compensate with three mechanisms:

1. **Temporal smoothing**: a sliding majority-vote window eliminates
   single-frame flickers before span construction.
2. **Span padding**: each merged span is expanded by ``onset_pad`` seconds
   at its start and ``offset_pad`` seconds at its end, then adjacent
   expanded spans are re-clamped so they do not overlap.
3. **Segment search tolerance**: ``speakers_for_segment`` accepts a
   ``boundary_tolerance`` argument that widens the lookup window at both
   ends of a transcript segment, so a segment that begins just *after* a
   speaker's span start is still attributed correctly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from .detector import ActiveSpeaker

log = logging.getLogger(__name__)


@dataclass
class SpeakerSpan:
    """A contiguous interval during which certain speakers were active."""
    start: float   # seconds
    end: float     # seconds
    speakers: list[str]

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        names = ", ".join(self.speakers) if self.speakers else "—"
        return f"SpeakerSpan({self.start:.1f}s–{self.end:.1f}s: {names})"


def _normalise_speakers(speakers: list[ActiveSpeaker]) -> list[str]:
    """Sort and deduplicate active speaker names."""
    seen: dict[str, float] = {}
    for s in speakers:
        name = s.name.strip()
        if name:
            if name not in seen or s.confidence > seen[name]:
                seen[name] = s.confidence
    return sorted(seen.keys())


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def _smooth_speakers(
    frame_speakers: list[list[str]],
    window: int = 3,
) -> list[list[str]]:
    """Apply a sliding majority-vote window to suppress single-frame flickers.

    For each frame, a speaker name must appear in at least half of the
    surrounding ``window`` frames to be kept.  This eliminates brief artefacts
    (e.g. one frame where the mic icon colour bleeds through the video codec)
    without shifting actual speaker transitions by more than half a window.
    """
    n = len(frame_speakers)
    if n == 0 or window <= 1:
        return frame_speakers

    half = window // 2
    smoothed: list[list[str]] = []

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_frames = frame_speakers[lo:hi]
        window_size = len(window_frames)

        counts: dict[str, int] = {}
        for speakers in window_frames:
            for s in speakers:
                counts[s] = counts.get(s, 0) + 1

        # Require majority support (> half the window) to survive smoothing
        threshold = window_size / 2.0
        winning = sorted(s for s, c in counts.items() if c > threshold)
        smoothed.append(winning)

    return smoothed


# ---------------------------------------------------------------------------
# Timeline builder
# ---------------------------------------------------------------------------

def build_timeline(
    frame_results: Sequence[tuple[float, list[ActiveSpeaker]]],
    *,
    frame_interval: float = 1.0,
    min_span_duration: float = 0.5,
    merge_gap: float = 2.0,
    onset_pad: float = 0.3,
    offset_pad: float = 0.2,
    smoothing_window: int = 3,
) -> list[SpeakerSpan]:
    """Convert per-frame speaker detections into a merged timeline.

    Args:
        frame_results: Sequence of (timestamp, [ActiveSpeaker]) pairs.
        frame_interval: Nominal seconds between sampled frames.
        min_span_duration: Discard spans shorter than this (noise).
        merge_gap: Merge consecutive same-speaker spans if gap ≤ this (seconds).
        onset_pad: Expand each span's *start* backwards by this many seconds
            to compensate for WebEx green-mic onset delay.
        offset_pad: Expand each span's *end* forwards by this many seconds
            to compensate for WebEx green-mic clearance delay.
        smoothing_window: Odd number of frames for majority-vote smoothing
            (1 = disabled).

    Returns:
        Sorted list of SpeakerSpan objects.
    """
    if not frame_results:
        return []

    # Extract speaker name lists per frame (sorted, deduped)
    frames_sorted = sorted(frame_results, key=lambda r: r[0])
    timestamps = [ts for ts, _ in frames_sorted]
    raw_speakers = [_normalise_speakers(active) for _, active in frames_sorted]

    # --- Temporal smoothing ---
    smoothed_speakers = _smooth_speakers(raw_speakers, window=smoothing_window)
    log.debug(
        "Timeline: %d frames, smoothing_window=%d", len(frames_sorted), smoothing_window
    )

    # --- Build raw per-frame spans ---
    raw: list[SpeakerSpan] = []
    for ts, names in zip(timestamps, smoothed_speakers):
        raw.append(SpeakerSpan(start=ts, end=ts + frame_interval, speakers=names))

    # --- Merge adjacent same-speaker spans ---
    merged: list[SpeakerSpan] = []
    for span in raw:
        if (
            merged
            and merged[-1].speakers == span.speakers
            and span.start - merged[-1].end <= merge_gap
        ):
            merged[-1].end = span.end
        else:
            merged.append(SpeakerSpan(
                start=span.start,
                end=span.end,
                speakers=list(span.speakers),
            ))

    # Drop short noise spans before padding (they would grow to onset_pad size)
    merged = [s for s in merged if s.duration >= min_span_duration]

    # --- Apply onset/offset padding ---
    if onset_pad > 0 or offset_pad > 0:
        for span in merged:
            span.start = max(0.0, span.start - onset_pad)
            span.end = span.end + offset_pad

        # Re-clamp: if expansion causes spans to overlap, pull the earlier span's
        # end back to the next span's start (preserving the later span's start).
        for i in range(len(merged) - 1):
            if merged[i].end > merged[i + 1].start:
                merged[i].end = merged[i + 1].start

        # Drop any spans that collapsed to zero or negative duration after clamping
        merged = [s for s in merged if s.duration > 0]

    log.info(
        "Timeline built: %d spans covering %d unique speaker(s)",
        len(merged),
        len({n for s in merged for n in s.speakers}),
    )
    return merged


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def speakers_at(timeline: list[SpeakerSpan], timestamp: float) -> list[str]:
    """Return the list of active speakers at a given timestamp."""
    for span in timeline:
        if span.start <= timestamp < span.end:
            return span.speakers
    return []


def speakers_for_segment(
    timeline: list[SpeakerSpan],
    start: float,
    end: float,
    *,
    boundary_tolerance: float = 0.4,
) -> list[str]:
    """Return the speaker(s) most active during a transcript segment [start, end).

    Args:
        timeline: Speaker activity spans.
        start: Segment start time (seconds).
        end: Segment end time (seconds).
        boundary_tolerance: Widen the lookup window by this many seconds at
            each boundary.  Compensates for residual timing drift between
            Whisper timestamps and the visual speaker timeline.

    Uses a weighted vote: each span contributes its overlap duration as weight.
    Returns all speakers within 80% of the top weight (captures crosstalk).
    """
    if not timeline:
        return ["Unknown"]

    # Widen the search window slightly at both ends
    search_start = start - boundary_tolerance
    search_end = end + boundary_tolerance

    weights: dict[str, float] = {}
    for span in timeline:
        overlap = min(search_end, span.end) - max(search_start, span.start)
        if overlap <= 0:
            continue
        # Weight by how much of the *original* segment window the span covers,
        # not the widened window — this keeps boundary_tolerance from inflating
        # scores for speakers who only overlap in the tolerance region.
        core_overlap = min(end, span.end) - max(start, span.start)
        tol_only = overlap - max(0.0, core_overlap)
        effective = max(0.0, core_overlap) + tol_only * 0.25  # discount tolerance region
        for name in span.speakers:
            weights[name] = weights.get(name, 0.0) + effective

    if not weights:
        return ["Unknown"]

    max_weight = max(weights.values())
    threshold = max_weight * 0.80
    return sorted(n for n, w in weights.items() if w >= threshold)
