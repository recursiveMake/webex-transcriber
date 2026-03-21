"""Align transcription segments to speaker timeline.

Produces annotated utterances: each utterance is one or more consecutive
Whisper segments attributed to the same speaker(s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from meeting_transcriber.transcription.whisper import Segment, TranscriptionResult
from meeting_transcriber.speaker.timeline import SpeakerSpan, speakers_for_segment


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Utterance:
    """A speaker-attributed speech unit."""
    speakers: list[str]
    text: str
    start: float   # seconds
    end: float     # seconds
    segments: list[Segment] = field(default_factory=list)

    @property
    def timestamp_str(self) -> str:
        return _fmt_time(self.start)

    @property
    def speaker_label(self) -> str:
        if not self.speakers or self.speakers == ["Unknown"]:
            return "Unknown"
        return " / ".join(self.speakers)

    def __repr__(self) -> str:
        return f"Utterance({self.speaker_label!r}, {self.start:.1f}–{self.end:.1f})"


# ---------------------------------------------------------------------------
# Aligner
# ---------------------------------------------------------------------------

def align(
    transcription: TranscriptionResult,
    timeline: list[SpeakerSpan],
    *,
    merge_gap: float = 1.5,
    boundary_tolerance: float = 0.4,
) -> list[Utterance]:
    """Attribute each Whisper segment to a speaker and merge into utterances.

    Args:
        transcription: Output of WhisperTranscriber.transcribe().
        timeline: Speaker activity spans from the visual detector.
        merge_gap: Consecutive segments from the same speaker(s) are merged
            if their gap is at most this many seconds.
        boundary_tolerance: Passed to ``speakers_for_segment``; widens the
            per-segment lookup window to absorb timing drift between Whisper
            and the visual speaker timeline.

    Returns:
        Sorted list of Utterance objects.
    """
    if not transcription.segments:
        return []

    # Assign speaker(s) to each segment
    attributed: list[tuple[list[str], Segment]] = []
    for seg in transcription.segments:
        names = speakers_for_segment(
            timeline, seg.start, seg.end,
            boundary_tolerance=boundary_tolerance,
        )
        attributed.append((names, seg))

    # Merge consecutive segments with the same speaker attribution
    utterances: list[Utterance] = []
    for names, seg in attributed:
        if (
            utterances
            and utterances[-1].speakers == names
            and seg.start - utterances[-1].end <= merge_gap
        ):
            utterances[-1].text += " " + seg.text.strip()
            utterances[-1].end = seg.end
            utterances[-1].segments.append(seg)
        else:
            utterances.append(Utterance(
                speakers=names,
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                segments=[seg],
            ))

    return utterances


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
