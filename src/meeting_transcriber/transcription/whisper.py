"""Whisper-based transcription using mlx-whisper (Apple Silicon native).

mlx-whisper leverages Apple's MLX framework for Metal-accelerated inference
on Apple Silicon — significantly faster than CPU-bound alternatives.

Produces word-level timestamps for precise speaker alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Word:
    """A single transcribed word with timing."""
    text: str
    start: float   # seconds
    end: float     # seconds
    confidence: float = 1.0

    def __repr__(self) -> str:
        return f"Word({self.text!r}, {self.start:.2f}–{self.end:.2f})"


@dataclass
class Segment:
    """A contiguous speech segment (sentence or clause)."""
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)
    language: str = "en"

    def __repr__(self) -> str:
        return f"Segment({self.start:.1f}–{self.end:.1f}: {self.text[:40]!r})"


@dataclass
class TranscriptionResult:
    """Full transcription output."""
    segments: list[Segment]
    language: str
    duration: float

    @property
    def words(self) -> list[Word]:
        return [w for seg in self.segments for w in seg.words]

    @property
    def full_text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments)


# ---------------------------------------------------------------------------
# Available model sizes (ordered by size)
# ---------------------------------------------------------------------------

MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
DEFAULT_MODEL = "large-v3"


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------

class WhisperTranscriber:
    """Wraps mlx-whisper for Apple Silicon optimised transcription.

    mlx-whisper uses temperature-based sampling, not beam search.
    At temperature=0 it is fully greedy (deterministic, fast).  The library
    automatically retries at higher temperatures if a segment's compression
    ratio or average log-probability indicates a likely hallucination.

    Args:
        model: Whisper model name (e.g. ``"large-v3"``, ``"medium"``).
        language: Force language (``"en"`` for English). None = auto-detect.
        temperature: Initial decoding temperature.  0.0 = greedy (default).
            Pass a tuple to override the full fallback schedule, e.g.
            ``(0.0, 0.2, 0.4)`` for a three-step schedule.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        language: str | None = "en",
        temperature: float | tuple[float, ...] = 0.0,
    ) -> None:
        self.model = model
        self.language = language
        self.temperature = temperature

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe a WAV file and return word-level timestamped results."""
        import mlx_whisper

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        repo = f"mlx-community/whisper-{self.model}-mlx"
        log.info(
            "Starting Whisper transcription: model=%s, temperature=%s, language=%s",
            self.model, self.temperature, self.language or "auto",
        )
        log.info("Model repo: %s (will download on first use)", repo)

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=repo,
            word_timestamps=True,
            language=self.language,
            temperature=self.temperature,
        )

        parsed = self._parse_result(result)
        log.info(
            "Transcription complete: %d segments, %.1fs audio, language=%s",
            len(parsed.segments), parsed.duration, parsed.language,
        )
        return parsed

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_result(self, raw: dict) -> TranscriptionResult:
        segments: list[Segment] = []
        language = raw.get("language", "en")

        for seg_raw in raw.get("segments", []):
            words = self._parse_words(seg_raw.get("words", []))
            seg = Segment(
                text=seg_raw.get("text", "").strip(),
                start=float(seg_raw.get("start", 0.0)),
                end=float(seg_raw.get("end", 0.0)),
                words=words,
                language=language,
            )
            if seg.text:
                segments.append(seg)

        # Use max end time rather than last segment end — Whisper segments are
        # usually ordered, but defensive against any reordering during parsing.
        duration = max((s.end for s in segments), default=0.0)
        return TranscriptionResult(segments=segments, language=language, duration=duration)

    def _parse_words(self, raw_words: list[dict]) -> list[Word]:
        words: list[Word] = []
        for w in raw_words:
            text = w.get("word", "").strip()
            if not text:
                continue
            words.append(Word(
                text=text,
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                confidence=float(w.get("probability", 1.0)),
            ))
        return words


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: Path,
    model: str = DEFAULT_MODEL,
    language: str | None = "en",
    temperature: float | tuple[float, ...] = 0.0,
) -> TranscriptionResult:
    """Transcribe audio and return word-level timestamped results."""
    return WhisperTranscriber(model=model, language=language, temperature=temperature).transcribe(audio_path)
