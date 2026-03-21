"""Tests for WhisperTranscriber.

Full transcription requires a real audio file and model download;
these tests cover data parsing and the result structure with mocked mlx_whisper.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from meeting_transcriber.transcription.whisper import (
    WhisperTranscriber,
    TranscriptionResult,
    Segment,
    Word,
    transcribe,
)


# ---------------------------------------------------------------------------
# Sample raw output (mimics mlx_whisper.transcribe return value)
# ---------------------------------------------------------------------------

_MOCK_RAW = {
    "language": "en",
    "segments": [
        {
            "start": 0.0,
            "end": 3.5,
            "text": " Hello everyone welcome to the meeting.",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.99},
                {"word": "everyone", "start": 0.5, "end": 1.0, "probability": 0.98},
                {"word": "welcome", "start": 1.0, "end": 1.6, "probability": 0.97},
                {"word": "to", "start": 1.6, "end": 1.8, "probability": 0.99},
                {"word": "the", "start": 1.8, "end": 2.0, "probability": 0.99},
                {"word": "meeting.", "start": 2.0, "end": 3.5, "probability": 0.95},
            ],
        },
        {
            "start": 4.0,
            "end": 7.0,
            "text": " Let's start with the agenda.",
            "words": [
                {"word": "Let's", "start": 4.0, "end": 4.5, "probability": 0.96},
                {"word": "start", "start": 4.5, "end": 5.0, "probability": 0.98},
                {"word": "with", "start": 5.0, "end": 5.3, "probability": 0.99},
                {"word": "the", "start": 5.3, "end": 5.5, "probability": 0.99},
                {"word": "agenda.", "start": 5.5, "end": 7.0, "probability": 0.97},
            ],
        },
    ],
}


class TestResultParsing:
    def setup_method(self):
        self.transcriber = WhisperTranscriber(model="tiny")
        self.result = self.transcriber._parse_result(_MOCK_RAW)

    def test_language_parsed(self):
        assert self.result.language == "en"

    def test_segment_count(self):
        assert len(self.result.segments) == 2

    def test_segment_text(self):
        assert "Hello everyone" in self.result.segments[0].text

    def test_segment_timing(self):
        assert self.result.segments[0].start == pytest.approx(0.0)
        assert self.result.segments[0].end == pytest.approx(3.5)

    def test_word_count(self):
        assert len(self.result.segments[0].words) == 6

    def test_word_attributes(self):
        word = self.result.segments[0].words[0]
        assert word.text == "Hello"
        assert word.start == pytest.approx(0.0)
        assert word.end == pytest.approx(0.5)
        assert word.confidence == pytest.approx(0.99)

    def test_full_text_property(self):
        text = self.result.full_text
        assert "Hello everyone" in text
        assert "agenda" in text

    def test_words_property_flattens(self):
        all_words = self.result.words
        assert len(all_words) == 11

    def test_duration(self):
        assert self.result.duration == pytest.approx(7.0)


class TestTranscriberMocked:
    def test_transcribe_calls_mlx_whisper(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"\x00" * 44)  # minimal file

        with patch("mlx_whisper.transcribe", return_value=_MOCK_RAW) as mock_fn:
            t = WhisperTranscriber(model="tiny", language="en")
            result = t.transcribe(wav)

        mock_fn.assert_called_once()
        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) == 2

    def test_transcribe_missing_file_raises(self):
        t = WhisperTranscriber(model="tiny")
        with pytest.raises(FileNotFoundError):
            t.transcribe(Path("/nonexistent/audio.wav"))

    def test_empty_segment_text_filtered(self):
        raw = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "", "words": []},
                {"start": 1.0, "end": 2.0, "text": "  ", "words": []},
            ],
        }
        t = WhisperTranscriber(model="tiny")
        result = t._parse_result(raw)
        assert result.segments == []

    def test_words_with_empty_text_filtered(self):
        raw = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0, "end": 2.0,
                    "text": "Hello",
                    "words": [
                        {"word": "", "start": 0.0, "end": 0.5, "probability": 0.9},
                        {"word": "Hello", "start": 0.5, "end": 1.0, "probability": 0.99},
                    ],
                }
            ],
        }
        t = WhisperTranscriber(model="tiny")
        result = t._parse_result(raw)
        assert len(result.segments[0].words) == 1
        assert result.segments[0].words[0].text == "Hello"


class TestConvenienceFunction:
    def test_transcribe_function(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"\x00" * 44)
        with patch("mlx_whisper.transcribe", return_value=_MOCK_RAW):
            result = transcribe(wav, model="tiny")
        assert isinstance(result, TranscriptionResult)
