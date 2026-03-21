"""Build a speaker activity timeline from per-frame detection results.

The timeline maps time intervals to lists of active speaker names.
Consecutive frames with the same speaker(s) are merged into spans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .detector import ActiveSpeaker


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


def build_timeline(
    frame_results: Sequence[tuple[float, list[ActiveSpeaker]]],
    *,
    frame_interval: float = 1.0,
    min_span_duration: float = 0.5,
    merge_gap: float = 2.0,
) -> list[SpeakerSpan]:
    """Convert per-frame speaker detections into a merged timeline.

    Args:
        frame_results: Sequence of (timestamp, [ActiveSpeaker]) pairs.
        frame_interval: Nominal seconds between sampled frames.
        min_span_duration: Discard spans shorter than this (noise).
        merge_gap: Merge consecutive spans from the same speaker(s) if the
            gap between them is at most this many seconds.

    Returns:
        Sorted list of SpeakerSpan objects.
    """
    if not frame_results:
        return []

    # Build raw spans: each frame creates one span of length = frame_interval
    raw: list[SpeakerSpan] = []
    for ts, active in frame_results:
        names = _normalise_speakers(active)
        span = SpeakerSpan(
            start=ts,
            end=ts + frame_interval,
            speakers=names,
        )
        raw.append(span)

    # Sort by start time
    raw.sort(key=lambda s: s.start)

    # Merge adjacent spans with matching speakers
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

    # Drop very short spans (typically noise)
    merged = [s for s in merged if s.duration >= min_span_duration]

    return merged


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
) -> list[str]:
    """Return the speaker(s) most active during a transcript segment [start, end).

    Uses a weighted vote: each span contributes its overlap duration as weight.
    Returns the names with the highest total overlap weight.
    """
    if not timeline:
        return ["Unknown"]

    weights: dict[str, float] = {}
    for span in timeline:
        overlap = min(end, span.end) - max(start, span.start)
        if overlap <= 0:
            continue
        for name in span.speakers:
            weights[name] = weights.get(name, 0.0) + overlap

    if not weights:
        return ["Unknown"]

    max_weight = max(weights.values())
    # Return all speakers within 80% of the max weight (handles crosstalk)
    threshold = max_weight * 0.80
    return sorted(n for n, w in weights.items() if w >= threshold)
