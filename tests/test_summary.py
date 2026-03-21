"""Tests for the summary generator.

Ollama calls are mocked so tests run without a running Ollama instance.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from meeting_transcriber.alignment.aligner import Utterance
from meeting_transcriber.summary.generator import (
    SummaryGenerator,
    _build_transcript_block,
    _build_prompt,
    summarise,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _utt(speaker: str, text: str, start: float, end: float) -> Utterance:
    return Utterance(speakers=[speaker], text=text, start=start, end=end)


SAMPLE_UTTERANCES = [
    _utt("Alice", "Welcome everyone to the weekly sync.", 0.0, 3.0),
    _utt("Bob", "Thanks Alice. I finished the API migration.", 4.0, 7.0),
    _utt("Alice", "Great. Let's make that official — Bob owns the API migration sign-off.", 8.0, 12.0),
    _utt("Carol", "I'll follow up on the testing by Friday.", 13.0, 16.0),
]


# ---------------------------------------------------------------------------
# Transcript block tests
# ---------------------------------------------------------------------------

class TestBuildTranscriptBlock:
    def test_contains_speaker_names(self):
        block = _build_transcript_block(SAMPLE_UTTERANCES)
        assert "Alice" in block
        assert "Bob" in block
        assert "Carol" in block

    def test_contains_timestamps(self):
        block = _build_transcript_block(SAMPLE_UTTERANCES)
        assert "00:00:00" in block
        assert "00:00:04" in block

    def test_truncation_applied_for_long_transcripts(self):
        long = [_utt("Alice", "word " * 200, float(i), float(i + 1)) for i in range(100)]
        block = _build_transcript_block(long, max_chars=500)
        assert "truncated" in block
        assert len(block) < 600  # a bit over due to truncation message

    def test_unknown_speaker_label(self):
        u = Utterance(speakers=["Unknown"], text="Something.", start=0.0, end=1.0)
        block = _build_transcript_block([u])
        assert "Unknown Speaker" in block


class TestBuildPrompt:
    def test_prompt_contains_transcript(self):
        prompt = _build_prompt(SAMPLE_UTTERANCES)
        assert "TRANSCRIPT START" in prompt
        assert "TRANSCRIPT END" in prompt
        assert "Alice" in prompt


# ---------------------------------------------------------------------------
# Generator tests (mocked Ollama)
# ---------------------------------------------------------------------------

_MOCK_RESPONSE = {
    "message": {
        "content": (
            "# Meeting Summary\n\n"
            "## Key Decisions\n- Bob owns API migration sign-off [Alice, 00:00:08]\n\n"
            "## Action Items\n- **Carol**: Follow up on testing by Friday [Carol, 00:00:13]\n\n"
            "## Topics Discussed\n- API migration status\n- Testing timeline\n"
        )
    }
}


class TestSummaryGenerator:
    def test_generates_markdown(self):
        with patch("ollama.chat", return_value=_MOCK_RESPONSE):
            gen = SummaryGenerator(model="test-model")
            summary = gen.generate(SAMPLE_UTTERANCES)
        assert "# Meeting Summary" in summary
        assert "Key Decisions" in summary
        assert "Action Items" in summary

    def test_empty_utterances_returns_placeholder(self):
        gen = SummaryGenerator()
        summary = gen.generate([])
        assert "No transcript content" in summary

    def test_ollama_error_raises_runtime_error(self):
        with patch("ollama.chat", side_effect=Exception("connection refused")):
            gen = SummaryGenerator(model="missing-model")
            with pytest.raises(RuntimeError, match="Ollama request failed"):
                gen.generate(SAMPLE_UTTERANCES)

    def test_available_models(self):
        mock_list = {"models": [{"name": "llama3.2"}, {"name": "gemma3:1b"}]}
        with patch("ollama.list", return_value=mock_list):
            gen = SummaryGenerator()
            models = gen.available_models()
        assert "llama3.2" in models
        assert "gemma3:1b" in models

    def test_available_models_on_error_returns_empty(self):
        with patch("ollama.list", side_effect=Exception("offline")):
            gen = SummaryGenerator()
            assert gen.available_models() == []


class TestSummariseFunction:
    def test_convenience_function(self):
        with patch("ollama.chat", return_value=_MOCK_RESPONSE):
            result = summarise(SAMPLE_UTTERANCES, model="test-model")
        assert "Meeting Summary" in result
