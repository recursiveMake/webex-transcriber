"""Generate a meeting summary with citations using a local Ollama model.

The summary includes:
- Key decisions made during the meeting
- Action items (owner + description)
- Topics discussed
- Inline citations in the form [Speaker, HH:MM:SS]
"""

from __future__ import annotations

import textwrap
from typing import Sequence

from meeting_transcriber.alignment.aligner import Utterance, _fmt_time


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a professional meeting summariser. Your job is to produce a
    concise, well-structured summary of a meeting transcript.

    Rules:
    - Write in clear, professional English.
    - Use markdown formatting with headers and bullet points.
    - Cite specific quotes or decisions with inline references using the
      format [Speaker Name, HH:MM:SS] immediately after the relevant sentence.
    - Include a "Key Decisions" section, an "Action Items" section, and a
      "Topics Discussed" section.
    - Keep action items in the format: **Owner**: description.
    - Do not invent content that is not in the transcript.
    - If the speaker is "Unknown", omit the speaker name from citations.
""")


def _build_transcript_block(utterances: Sequence[Utterance], max_chars: int = 12000) -> str:
    """Format utterances as a plain text transcript block for the prompt."""
    lines: list[str] = []
    for u in utterances:
        label = u.speaker_label if u.speaker_label != "Unknown" else "Unknown Speaker"
        lines.append(f"[{u.timestamp_str}] {label}: {u.text}")

    block = "\n".join(lines)
    if len(block) > max_chars:
        # Truncate from the middle to preserve start and end context
        half = max_chars // 2
        block = block[:half] + "\n...[transcript truncated]...\n" + block[-half:]
    return block


def _build_prompt(utterances: Sequence[Utterance]) -> str:
    transcript = _build_transcript_block(utterances)
    return (
        "Below is the transcript of a meeting. "
        "Please produce a structured summary following the rules you have been given.\n\n"
        "---TRANSCRIPT START---\n"
        + transcript
        + "\n---TRANSCRIPT END---\n\n"
        "Write the summary now:"
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SummaryGenerator:
    """Generates meeting summaries via a local Ollama model.

    Args:
        model: Ollama model name (e.g. ``"llama3.2"``, ``"mistral"``,
            ``"gemma3:1b"``). Must be pulled locally via ``ollama pull``.
        temperature: Sampling temperature (lower = more deterministic).
        context_window: Maximum tokens to request from Ollama.
    """

    def __init__(
        self,
        model: str = "gemma3:1b",
        temperature: float = 0.2,
        context_window: int = 8192,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.context_window = context_window

    def generate(self, utterances: Sequence[Utterance]) -> str:
        """Generate a markdown summary from a list of utterances.

        Returns the summary as a markdown string.
        Raises ``RuntimeError`` if Ollama is unreachable or the model is not pulled.
        """
        import ollama

        if not utterances:
            return "# Meeting Summary\n\n*No transcript content available.*\n"

        prompt = _build_prompt(utterances)

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.context_window,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed (model={self.model!r}). "
                f"Ensure Ollama is running and the model is pulled.\n"
                f"  ollama pull {self.model}\nError: {exc}"
            ) from exc

        # SDK may return a ChatResponse object or a dict depending on version
        msg = getattr(response, "message", None)
        if msg is not None:
            return getattr(msg, "content", str(msg))
        return response["message"]["content"]

    def available_models(self) -> list[str]:
        """Return a list of locally available Ollama model names."""
        import ollama
        try:
            response = ollama.list()
            # SDK returns a ListResponse with .models (list of Model objects)
            # Older versions returned a dict with "models" key
            models = getattr(response, "models", None) or response.get("models", [])
            return [getattr(m, "model", None) or m.get("name", "") for m in models]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def summarise(
    utterances: Sequence[Utterance],
    model: str = "gemma3:1b",
    temperature: float = 0.2,
) -> str:
    """Generate a markdown summary from utterances using a local Ollama model."""
    return SummaryGenerator(model=model, temperature=temperature).generate(utterances)
