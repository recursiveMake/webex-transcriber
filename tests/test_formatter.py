"""Tests for the Markdown output formatter."""

from __future__ import annotations

from pathlib import Path

import pytest

from meeting_transcriber.alignment.aligner import Utterance
from meeting_transcriber.output.formatter import (
    format_transcript,
    format_summary,
    write_outputs,
)


def _utt(speaker: str, text: str, start: float, end: float) -> Utterance:
    return Utterance(speakers=[speaker], text=text, start=start, end=end)


UTTERANCES = [
    _utt("Alice", "Welcome everyone.", 0.0, 2.0),
    _utt("Bob", "Thanks Alice. Ready to start.", 3.0, 6.0),
    _utt("Alice", "Let's begin with the agenda.", 7.0, 10.0),
]


class TestFormatTranscript:
    def test_has_header(self):
        md = format_transcript(UTTERANCES)
        assert "# Meeting Transcript" in md

    def test_contains_speaker_names(self):
        md = format_transcript(UTTERANCES)
        assert "Alice" in md
        assert "Bob" in md

    def test_contains_timestamps(self):
        md = format_transcript(UTTERANCES)
        assert "00:00:00" in md
        assert "00:00:03" in md

    def test_speaker_block_header_format(self):
        md = format_transcript(UTTERANCES)
        assert "**[00:00:00] Alice**" in md

    def test_source_filename_included(self):
        md = format_transcript(UTTERANCES, video_path=Path("/tmp/weekly_sync.mp4"))
        assert "weekly_sync.mp4" in md

    def test_empty_utterances(self):
        md = format_transcript([])
        assert "# Meeting Transcript" in md

    def test_unknown_speaker(self):
        u = Utterance(speakers=["Unknown"], text="Someone said something.", start=0.0, end=1.0)
        md = format_transcript([u])
        assert "Unknown" in md


class TestFormatSummary:
    def test_has_wrapper_header(self):
        md = format_summary("## Key Decisions\n- Something was decided.\n")
        assert "# Meeting Summary" in md

    def test_deduplicates_header(self):
        body = "# Meeting Summary\n\n## Key Decisions\n- Decision.\n"
        md = format_summary(body)
        # Should not have two "# Meeting Summary" headers
        count = md.count("# Meeting Summary")
        assert count == 1

    def test_source_included(self):
        md = format_summary("content", video_path=Path("/tmp/meeting.mp4"))
        assert "meeting.mp4" in md

    def test_body_preserved(self):
        body = "## Action Items\n- **Alice**: Follow up.\n"
        md = format_summary(body)
        assert "Action Items" in md
        assert "Alice" in md


class TestWriteOutputs:
    def test_creates_files(self, tmp_path):
        summary = "## Key Decisions\n- Something.\n"
        t_path, s_path = write_outputs(UTTERANCES, summary, tmp_path, stem="test")
        assert t_path.exists()
        assert s_path.exists()

    def test_file_names(self, tmp_path):
        t_path, s_path = write_outputs(UTTERANCES, "summary", tmp_path, stem="weekly")
        assert t_path.name == "weekly_transcript.md"
        assert s_path.name == "weekly_summary.md"

    def test_transcript_content(self, tmp_path):
        t_path, _ = write_outputs(UTTERANCES, "summary", tmp_path)
        content = t_path.read_text()
        assert "Alice" in content

    def test_summary_content(self, tmp_path):
        _, s_path = write_outputs(UTTERANCES, "## Key Decisions\n- D.\n", tmp_path)
        content = s_path.read_text()
        assert "Key Decisions" in content
