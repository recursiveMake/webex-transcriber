"""Integration test: run the full pipeline on a synthetic WebEx video.

Marked as slow (requires ffmpeg, mlx-whisper model download, Ollama).
Run with:  pytest tests/test_integration.py -v -m integration
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.fixtures.make_test_video import make_test_video


# All tests in this file are integration tests — skip unless explicitly requested
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("integration")
    video = tmp / "synthetic_meeting.mp4"
    make_test_video(video, duration=12.0, fps=5)
    return video


class TestPipelineEndToEnd:
    def test_video_created(self, synthetic_video):
        assert synthetic_video.exists()
        assert synthetic_video.stat().st_size > 0

    def test_extractor_can_probe(self, synthetic_video):
        from meeting_transcriber.video.extractor import VideoExtractor
        ex = VideoExtractor(synthetic_video)
        assert 10 < ex.duration < 15
        assert ex.resolution[0] > 0

    def test_speaker_detection_finds_active_speakers(self, synthetic_video, tmp_path):
        import cv2
        from meeting_transcriber.video.extractor import VideoExtractor
        from meeting_transcriber.speaker.layout import find_mic_blobs

        ex = VideoExtractor(synthetic_video, frame_interval=1.0)
        _, frames = ex.extract_all(tmp_path / "work")

        found_green = False
        for ts, frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is not None and find_mic_blobs(frame):
                found_green = True
                break
        assert found_green, "No green mic blobs found in any sampled frame"

    def test_full_pipeline_with_mocked_whisper_and_ollama(self, synthetic_video, tmp_path):
        """Run the full pipeline with mocked transcription and summary."""
        from meeting_transcriber.pipeline import PipelineConfig, run
        from meeting_transcriber.transcription.whisper import TranscriptionResult, Segment

        mock_transcription = TranscriptionResult(
            segments=[
                Segment(text="Hello everyone.", start=0.0, end=3.0, words=[]),
                Segment(text="Thanks for joining.", start=4.0, end=7.0, words=[]),
                Segment(text="Let us review the agenda.", start=7.5, end=11.0, words=[]),
            ],
            language="en",
            duration=11.0,
        )
        mock_summary = (
            "## Key Decisions\n- None recorded.\n\n"
            "## Action Items\n- None recorded.\n\n"
            "## Topics Discussed\n- Agenda review.\n"
        )

        with (
            patch(
                "meeting_transcriber.transcription.whisper.WhisperTranscriber.transcribe",
                return_value=mock_transcription,
            ),
            patch("ollama.chat", return_value=SimpleNamespace(
                message=SimpleNamespace(content=mock_summary)
            )),
        ):
            config = PipelineConfig(
                whisper_model="tiny",
                ollama_model="gemma3:1b",
                frame_interval=2.0,
                output_dir=tmp_path,
            )
            result = run(synthetic_video, config=config)

        assert result.transcript_path.exists()
        assert result.summary_path.exists()

        transcript = result.transcript_path.read_text()
        assert "# Meeting Transcript" in transcript
        # At least one speaker should be identified from the synthetic frames
        assert len(result.utterances) > 0

        summary = result.summary_path.read_text()
        assert "# Meeting Summary" in summary
