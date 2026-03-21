"""Tests for the transcript-to-speaker alignment module."""

from __future__ import annotations

import pytest

from meeting_transcriber.alignment.aligner import Utterance, align, _fmt_time
from meeting_transcriber.speaker.timeline import SpeakerSpan
from meeting_transcriber.transcription.whisper import Segment, TranscriptionResult, Word


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(text: str, start: float, end: float) -> Segment:
    return Segment(text=text, start=start, end=end, words=[], language="en")


def _result(*segments: Segment) -> TranscriptionResult:
    return TranscriptionResult(
        segments=list(segments),
        language="en",
        duration=segments[-1].end if segments else 0.0,
    )


def _span(start: float, end: float, *names: str) -> SpeakerSpan:
    return SpeakerSpan(start=start, end=end, speakers=list(names))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAlign:
    def test_empty_transcription(self):
        result = _result()
        utterances = align(result, [])
        assert utterances == []

    def test_single_segment_known_speaker(self):
        result = _result(_seg("Hello world.", 0.0, 2.0))
        timeline = [_span(0.0, 5.0, "Alice")]
        utterances = align(result, timeline)
        assert len(utterances) == 1
        assert utterances[0].speakers == ["Alice"]
        assert "Hello world" in utterances[0].text

    def test_single_segment_no_speaker_is_unknown(self):
        result = _result(_seg("Hello world.", 0.0, 2.0))
        utterances = align(result, [])
        assert utterances[0].speakers == ["Unknown"]

    def test_consecutive_same_speaker_merged(self):
        result = _result(
            _seg("First sentence.", 0.0, 2.0),
            _seg("Second sentence.", 2.1, 4.0),
        )
        timeline = [_span(0.0, 10.0, "Alice")]
        utterances = align(result, timeline, merge_gap=0.5)
        assert len(utterances) == 1
        assert "First sentence" in utterances[0].text
        assert "Second sentence" in utterances[0].text

    def test_speaker_change_splits_utterance(self):
        result = _result(
            _seg("Alice speaks.", 0.0, 2.0),
            _seg("Bob speaks.", 3.0, 5.0),
        )
        timeline = [
            _span(0.0, 2.5, "Alice"),
            _span(2.5, 6.0, "Bob"),
        ]
        utterances = align(result, timeline)
        assert len(utterances) == 2
        assert utterances[0].speakers == ["Alice"]
        assert utterances[1].speakers == ["Bob"]

    def test_segment_spanning_two_speakers_assigns_dominant(self):
        # Segment runs 0–5; Alice active 0–1, Bob active 1–5 → Bob dominant
        result = _result(_seg("Mostly Bob's words.", 0.0, 5.0))
        timeline = [
            _span(0.0, 1.0, "Alice"),
            _span(1.0, 5.0, "Bob"),
        ]
        utterances = align(result, timeline)
        assert "Bob" in utterances[0].speakers

    def test_utterance_timestamps(self):
        result = _result(_seg("Hello.", 65.0, 67.5))
        timeline = [_span(60.0, 70.0, "Alice")]
        utterances = align(result, timeline)
        assert utterances[0].start == pytest.approx(65.0)
        assert utterances[0].end == pytest.approx(67.5)
        assert utterances[0].timestamp_str == "00:01:05"

    def test_speaker_label_unknown(self):
        u = Utterance(speakers=["Unknown"], text="...", start=0.0, end=1.0)
        assert u.speaker_label == "Unknown"

    def test_speaker_label_multiple(self):
        u = Utterance(speakers=["Alice", "Bob"], text="...", start=0.0, end=1.0)
        assert "Alice" in u.speaker_label
        assert "Bob" in u.speaker_label


class TestFmtTime:
    def test_seconds_only(self):
        assert _fmt_time(45) == "00:00:45"

    def test_minutes_and_seconds(self):
        assert _fmt_time(90) == "00:01:30"

    def test_hours(self):
        assert _fmt_time(3661) == "01:01:01"
