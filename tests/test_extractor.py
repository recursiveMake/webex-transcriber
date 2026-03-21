"""Tests for VideoExtractor.

Integration tests require a real video file; unit tests use synthetic fixtures.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meeting_transcriber.video.extractor import VideoExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_video(path: Path, duration: float = 3.0) -> Path:
    """Create a tiny synthetic MP4 using ffmpeg (no camera needed)."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=blue:size=320x240:rate=1:duration={duration}",
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("video")
    video_path = tmp / "test.mp4"
    _make_tiny_video(video_path, duration=3.0)
    return video_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVideoExtractorProbe:
    def test_duration(self, tiny_video):
        ex = VideoExtractor(tiny_video)
        assert 2.5 < ex.duration < 4.0

    def test_fps(self, tiny_video):
        ex = VideoExtractor(tiny_video)
        assert ex.fps == pytest.approx(1.0, abs=0.5)

    def test_resolution(self, tiny_video):
        ex = VideoExtractor(tiny_video)
        assert ex.resolution == (320, 240)


class TestAudioExtraction:
    def test_extract_audio_creates_wav(self, tiny_video, tmp_path):
        ex = VideoExtractor(tiny_video)
        audio = ex.extract_audio(tmp_path / "audio.wav")
        assert audio.exists()
        assert audio.stat().st_size > 0

    def test_extract_audio_is_16khz_mono(self, tiny_video, tmp_path):
        ex = VideoExtractor(tiny_video)
        audio = ex.extract_audio(tmp_path / "audio.wav")
        # Check with ffprobe
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(audio)],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        assert int(stream["sample_rate"]) == 16000
        assert int(stream["channels"]) == 1


class TestFrameExtraction:
    def test_extract_frames_count(self, tiny_video, tmp_path):
        ex = VideoExtractor(tiny_video, frame_interval=1.0)
        frames = ex.extract_frames(tmp_path / "frames")
        # 3-second video at 1fps → 3 frames
        assert 2 <= len(frames) <= 4

    def test_extract_frames_timestamps_increase(self, tiny_video, tmp_path):
        ex = VideoExtractor(tiny_video, frame_interval=1.0)
        frames = ex.extract_frames(tmp_path / "frames")
        timestamps = [t for t, _ in frames]
        assert timestamps == sorted(timestamps)

    def test_extract_frames_files_exist(self, tiny_video, tmp_path):
        ex = VideoExtractor(tiny_video, frame_interval=1.0)
        frames = ex.extract_frames(tmp_path / "frames")
        for _, path in frames:
            assert path.exists()


class TestExtractAll:
    def test_extract_all_parallel(self, tiny_video, tmp_path):
        ex = VideoExtractor(tiny_video, frame_interval=1.0)
        audio, frames = ex.extract_all(tmp_path / "work")
        assert audio.exists()
        assert len(frames) >= 2
