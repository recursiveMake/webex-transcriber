"""Tests for speaker layout detection."""

from __future__ import annotations

import numpy as np
import pytest

from meeting_transcriber.speaker.layout import (
    LayoutTracker,
    detect_layout,
    find_mic_blobs,
    infer_tile_from_mic,
)
from tests.fixtures.synthetic import (
    no_active_speaker_frame,
    two_participant_frame,
)


class TestFindMicBlobs:
    def test_finds_green_blob(self):
        frame = two_participant_frame(active="Alice")
        blobs = find_mic_blobs(frame)
        assert len(blobs) >= 1

    def test_no_blobs_when_none_active(self):
        frame = no_active_speaker_frame()
        blobs = find_mic_blobs(frame)
        assert len(blobs) == 0

    def test_two_active_speakers_two_blobs(self):
        from tests.fixtures.synthetic import make_webex_frame, FRAME_W, FRAME_H, _TILE_W, _TILE_H, _RIGHT_X
        frame = make_webex_frame([
            {"name": "Alice", "tile": (_RIGHT_X, 10, _TILE_W, _TILE_H), "active": True},
            {"name": "Bob", "tile": (_RIGHT_X, 10 + _TILE_H + 8, _TILE_W, _TILE_H), "active": True},
        ])
        blobs = find_mic_blobs(frame)
        assert len(blobs) == 2


class TestInferTile:
    def test_tile_within_frame(self):
        blob = (100, 100, 10, 10)
        tile = infer_tile_from_mic(blob, frame_h=720, frame_w=1280)
        assert tile.x >= 0
        assert tile.y >= 0
        assert tile.x + tile.w <= 1280
        assert tile.y + tile.h <= 720

    def test_tile_covers_mic(self):
        blob = (200, 300, 12, 12)
        tile = infer_tile_from_mic(blob, frame_h=720, frame_w=1280)
        cx, cy = 206, 306
        assert tile.x <= cx <= tile.x + tile.w
        assert tile.y <= cy <= tile.y + tile.h


class TestDetectLayout:
    def test_detects_one_active_speaker(self):
        frame = two_participant_frame(active="Alice")
        snap = detect_layout([(0.0, frame)], min_tile_frames=1)
        assert snap is not None
        assert len(snap.tiles) >= 1

    def test_no_layout_when_no_green(self):
        frame = no_active_speaker_frame()
        snap = detect_layout([(0.0, frame)], min_tile_frames=1)
        assert snap is None or len(snap.tiles) == 0

    def test_multi_frame_stability(self):
        frames = [(float(i), two_participant_frame(active="Alice")) for i in range(5)]
        snap = detect_layout(frames, min_tile_frames=3)
        assert snap is not None
        assert len(snap.tiles) == 1


class TestLayoutTracker:
    def test_tracker_accumulates_frames(self):
        tracker = LayoutTracker(window=5)
        frame = two_participant_frame(active="Alice")
        result = None
        for i in range(6):
            result = tracker.update(float(i), frame)
        assert result is not None
        assert len(result.tiles) >= 1
