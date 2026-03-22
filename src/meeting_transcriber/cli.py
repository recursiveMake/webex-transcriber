"""Command-line interface for meeting-transcriber."""

from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from meeting_transcriber.logging_config import setup as setup_logging
from meeting_transcriber.pipeline import PipelineConfig, run
from meeting_transcriber.transcription.whisper import MODELS, DEFAULT_MODEL

console = Console()
err_console = Console(stderr=True)


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for output files (default: same directory as video).",
)
@click.option(
    "--whisper-model", "-m",
    type=click.Choice(MODELS),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Whisper model to use for transcription.",
)
@click.option(
    "--ollama-model",
    default="gemma3:1b",
    show_default=True,
    help="Ollama model for summary generation (must be pulled locally).",
)
@click.option(
    "--frame-interval",
    type=float,
    default=0.5,
    show_default=True,
    help="Seconds between sampled frames for speaker detection.",
)
@click.option(
    "--keep-work-dir",
    is_flag=True,
    default=False,
    help="Keep the temporary working directory after processing.",
)
@click.option(
    "--list-ollama-models",
    is_flag=True,
    default=False,
    help="List locally available Ollama models and exit.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable DEBUG logging (very detailed output).",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Suppress INFO logs; show only warnings and errors.",
)
def main(
    video: Path,
    output_dir: Path | None,
    whisper_model: str,
    ollama_model: str,
    frame_interval: float,
    keep_work_dir: bool,
    list_ollama_models: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Transcribe and summarise a WebEx meeting video.

    VIDEO is the path to an .mp4 recording.

    Outputs two Markdown files:
      - <stem>_transcript.md  — timestamped, speaker-attributed transcript
      - <stem>_summary.md     — structured meeting summary with citations
    """
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level, quiet=quiet)

    if list_ollama_models:
        from meeting_transcriber.summary.generator import SummaryGenerator
        models = SummaryGenerator().available_models()
        if models:
            console.print("[bold]Locally available Ollama models:[/bold]")
            for m in models:
                console.print(f"  • {m}")
        else:
            console.print("[yellow]No Ollama models found. Is Ollama running?[/yellow]")
        return

    if video is None:
        raise click.UsageError("VIDEO argument is required unless --list-ollama-models is used.")

    resolved_output = output_dir or video.parent

    console.print(Panel.fit(
        f"[bold cyan]Meeting Transcriber[/bold cyan]\n"
        f"Video:          {video.name}\n"
        f"Whisper model:  {whisper_model}\n"
        f"Ollama model:   {ollama_model}\n"
        f"Output dir:     {resolved_output}",
        border_style="cyan",
    ))

    config = PipelineConfig(
        whisper_model=whisper_model,
        ollama_model=ollama_model,
        frame_interval=frame_interval,
        output_dir=resolved_output,
        keep_work_dir=keep_work_dir,
    )

    try:
        result = run(video, config=config)
    except RuntimeError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    # ----------------------------------------------------------------
    # Print result summary
    # ----------------------------------------------------------------
    table = Table(title="Processing Complete", border_style="green")
    table.add_column("Item", style="bold")
    table.add_column("Value")

    table.add_row("Participants detected", str(len(result.participants)))
    table.add_row("Participants", ", ".join(result.participants) or "—")
    table.add_row("Transcript segments", str(len(result.utterances)))
    table.add_row("Timeline spans", str(len(result.timeline)))
    table.add_row("Transcript", str(result.transcript_path))
    table.add_row("Summary", str(result.summary_path))

    console.print(table)


if __name__ == "__main__":
    main()
