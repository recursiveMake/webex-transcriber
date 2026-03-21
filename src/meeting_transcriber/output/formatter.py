"""Format aligned utterances as a Markdown transcript document."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from meeting_transcriber.alignment.aligner import Utterance, _fmt_time


def format_transcript(
    utterances: Sequence[Utterance],
    video_path: Path | str | None = None,
) -> str:
    """Render utterances as a Markdown transcript.

    Each utterance becomes:

        **[HH:MM:SS] Speaker Name**
        Text of the utterance.

    Consecutive utterances from the same speaker with very short gaps
    are separated by a blank line rather than a new header.
    """
    lines: list[str] = []

    # Header
    lines.append("# Meeting Transcript\n")
    if video_path:
        lines.append(f"*Source: {Path(video_path).name}*\n")
    lines.append("---\n")

    prev_speaker: str | None = None
    for u in utterances:
        label = u.speaker_label
        if label != prev_speaker:
            if prev_speaker is not None:
                lines.append("")  # blank line between speaker blocks
            lines.append(f"**[{u.timestamp_str}] {label}**")
            prev_speaker = label
        lines.append(u.text)

    return "\n".join(lines) + "\n"


def format_summary(
    summary_md: str,
    video_path: Path | str | None = None,
) -> str:
    """Wrap the LLM-generated summary in a standard document header."""
    header_lines: list[str] = ["# Meeting Summary\n"]
    if video_path:
        header_lines.append(f"*Source: {Path(video_path).name}*\n")
    header_lines.append("---\n")
    header = "\n".join(header_lines)

    # Avoid double "# Meeting Summary" header if the LLM included one
    body = summary_md.strip()
    if body.lower().startswith("# meeting summary"):
        # Remove the first line
        body = "\n".join(body.splitlines()[1:]).lstrip()

    return header + body + "\n"


def write_outputs(
    utterances: Sequence[Utterance],
    summary_md: str,
    output_dir: Path,
    stem: str = "meeting",
    video_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write transcript.md and summary.md to output_dir.

    Returns (transcript_path, summary_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = output_dir / f"{stem}_transcript.md"
    summary_path = output_dir / f"{stem}_summary.md"

    transcript_path.write_text(
        format_transcript(utterances, video_path=video_path),
        encoding="utf-8",
    )
    summary_path.write_text(
        format_summary(summary_md, video_path=video_path),
        encoding="utf-8",
    )

    return transcript_path, summary_path
