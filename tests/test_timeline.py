"""Tests for speaker timeline building."""

from __future__ import annotations

import pytest

from meeting_transcriber.speaker.detector import ActiveSpeaker
from meeting_transcriber.speaker.layout import TileRegion
from meeting_transcriber.speaker.timeline import (
    SpeakerSpan,
    build_timeline,
    speakers_at,
    speakers_for_segment,
)


def _make_speaker(name: str) -> ActiveSpeaker:
    tile = TileRegion(x=0, y=0, w=100, h=80, mic_x=90, mic_y=70)
    return ActiveSpeaker(name=name, tile=tile)


class TestBuildTimeline:
    def test_empty_input(self):
        assert build_timeline([]) == []

    def test_single_frame(self):
        results = [(0.0, [_make_speaker("Alice")])]
        spans = build_timeline(results, frame_interval=1.0, min_span_duration=0.0)
        assert len(spans) == 1
        assert spans[0].speakers == ["Alice"]

    def test_consecutive_frames_merged(self):
        results = [
            (0.0, [_make_speaker("Alice")]),
            (1.0, [_make_speaker("Alice")]),
            (2.0, [_make_speaker("Alice")]),
        ]
        spans = build_timeline(results, frame_interval=1.0, min_span_duration=0.0)
        assert len(spans) == 1
        assert spans[0].start == 0.0
        assert spans[0].end == pytest.approx(3.0, abs=0.1)

    def test_speaker_change_creates_new_span(self):
        results = [
            (0.0, [_make_speaker("Alice")]),
            (1.0, [_make_speaker("Bob")]),
        ]
        spans = build_timeline(results, frame_interval=1.0, min_span_duration=0.0)
        assert len(spans) == 2
        assert spans[0].speakers == ["Alice"]
        assert spans[1].speakers == ["Bob"]

    def test_empty_frame_breaks_span(self):
        results = [
            (0.0, [_make_speaker("Alice")]),
            (1.0, []),
            (2.0, [_make_speaker("Alice")]),
        ]
        spans = build_timeline(results, frame_interval=1.0, min_span_duration=0.0, merge_gap=0.5)
        assert any(s.speakers == [] for s in spans)


class TestSpeakersAt:
    def setup_method(self):
        self.timeline = [
            SpeakerSpan(start=0.0, end=5.0, speakers=["Alice"]),
            SpeakerSpan(start=5.0, end=10.0, speakers=["Bob"]),
        ]

    def test_lookup_in_first_span(self):
        assert speakers_at(self.timeline, 2.0) == ["Alice"]

    def test_lookup_at_boundary(self):
        assert speakers_at(self.timeline, 5.0) == ["Bob"]

    def test_lookup_after_all_spans(self):
        assert speakers_at(self.timeline, 15.0) == []


class TestSpeakersForSegment:
    def setup_method(self):
        self.timeline = [
            SpeakerSpan(start=0.0, end=5.0, speakers=["Alice"]),
            SpeakerSpan(start=5.0, end=10.0, speakers=["Bob"]),
        ]

    def test_entirely_within_one_span(self):
        names = speakers_for_segment(self.timeline, 1.0, 3.0)
        assert names == ["Alice"]

    def test_segment_spanning_two_speakers_returns_dominant(self):
        # 1 second of Alice, 4 seconds of Bob → Bob dominant
        names = speakers_for_segment(self.timeline, 4.0, 9.0)
        assert "Bob" in names

    def test_no_overlap_returns_unknown(self):
        names = speakers_for_segment(self.timeline, 20.0, 25.0)
        assert names == ["Unknown"]

    def test_empty_timeline_returns_unknown(self):
        assert speakers_for_segment([], 0.0, 5.0) == ["Unknown"]
