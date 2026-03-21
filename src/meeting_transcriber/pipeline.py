"""Main processing pipeline with parallel execution.

Architecture:

    ┌── extract_audio ──────────────────► whisper ──────────────┐
    │                                                            ├──► align ──► summarise ──► write
    └── extract_frames ──► speaker_detection (parallel) ─────────┘

Audio extraction + frame extraction run in parallel (ThreadPoolExecutor).
Whisper runs concurrently with speaker frame-processing (ProcessPoolExecutor-
style via threads since mlx-whisper/EasyOCR both hold the GIL minimally).
"""

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from meeting_transcriber.alignment.aligner import Utterance, align
from meeting_transcriber.output.formatter import write_outputs
from meeting_transcriber.speaker.detector import SpeakerDetector
from meeting_transcriber.speaker.layout import LayoutTracker
from meeting_transcriber.speaker.timeline import SpeakerSpan, build_timeline
from meeting_transcriber.summary.generator import SummaryGenerator
from meeting_transcriber.transcription.whisper import TranscriptionResult, WhisperTranscriber
from meeting_transcriber.video.extractor import VideoExtractor


console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Tunable parameters for the processing pipeline."""

    # Whisper
    whisper_model: str = "large-v3"
    whisper_beam_size: int = 5

    # Frame sampling
    frame_interval: float = 1.0       # seconds between sampled frames

    # Speaker detection
    layout_window: int = 10           # frames for layout accumulation
    layout_drift_threshold: int = 80  # px drift before re-calibration
    mic_cluster_radius: int = 40      # px radius for blob deduplication

    # Timeline
    timeline_merge_gap: float = 2.0   # seconds; merge same-speaker gaps
    min_span_duration: float = 0.5    # drop spans shorter than this

    # Alignment
    alignment_merge_gap: float = 1.5  # seconds; merge same-speaker segments

    # Summary
    ollama_model: str = "gemma3:1b"
    ollama_temperature: float = 0.2

    # Output
    output_dir: Path = Path(".")
    keep_work_dir: bool = False


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    utterances: list[Utterance]
    timeline: list[SpeakerSpan]
    transcription: TranscriptionResult
    transcript_path: Path
    summary_path: Path
    participants: list[str]


# ---------------------------------------------------------------------------
# Speaker detection worker (runs in thread, processes all frames)
# ---------------------------------------------------------------------------

def _run_speaker_detection(
    frames: list[tuple[float, Path]],
    config: PipelineConfig,
    progress_callback: Callable[[int], None] | None = None,
) -> list[tuple[float, list]]:
    """Process all sampled frames and return per-frame speaker detections."""
    tracker = LayoutTracker(
        window=config.layout_window,
        drift_threshold=config.layout_drift_threshold,
    )
    detector = SpeakerDetector(mic_cluster_radius=config.mic_cluster_radius)

    results: list[tuple[float, list]] = []

    for i, (ts, frame_path) in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        layout = tracker.update(ts, frame)
        layout_tiles = layout.tiles if layout else None
        active = detector.process_frame(frame, layout_tiles=layout_tiles)
        results.append((ts, active))

        if progress_callback:
            progress_callback(i + 1)

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    video_path: Path,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the full meeting transcription pipeline.

    Args:
        video_path: Path to the input MP4 file.
        config: Pipeline configuration (uses defaults if None).

    Returns:
        PipelineResult with paths to generated Markdown files.
    """
    if config is None:
        config = PipelineConfig()

    video_path = Path(video_path)
    stem = video_path.stem

    work_dir = Path(tempfile.mkdtemp(prefix="meeting_work_"))
    console.log(f"[dim]Work directory: {work_dir}[/dim]")

    try:
        _run_pipeline(video_path, stem, work_dir, config)
        # Read the final result back
        result = _cached_result
    finally:
        if not config.keep_work_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

    return result


# We use a module-level variable to pass the result out of the nested function
# (avoids complex return-value threading across concurrent futures).
_cached_result: PipelineResult | None = None


def _run_pipeline(
    video_path: Path,
    stem: str,
    work_dir: Path,
    config: PipelineConfig,
) -> None:
    global _cached_result

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:

        # ----------------------------------------------------------------
        # Stage 1: Extract audio + frames in parallel
        # ----------------------------------------------------------------
        extract_task = progress.add_task("[cyan]Extracting audio & frames…", total=None)
        extractor = VideoExtractor(video_path, frame_interval=config.frame_interval)
        audio_path, frames = extractor.extract_all(work_dir)
        total_frames = len(frames)
        progress.update(extract_task, completed=1, total=1,
                        description=f"[green]Extracted {total_frames} frames + audio")

        # ----------------------------------------------------------------
        # Stage 2: Whisper + speaker detection in parallel
        # ----------------------------------------------------------------
        whisper_task = progress.add_task("[cyan]Transcribing audio (Whisper)…", total=None)
        speaker_task = progress.add_task("[cyan]Detecting speakers…", total=total_frames)

        transcription_result: TranscriptionResult | None = None
        frame_results: list[tuple[float, list]] | None = None

        def _transcribe():
            nonlocal transcription_result
            t = WhisperTranscriber(
                model=config.whisper_model,
                beam_size=config.whisper_beam_size,
            )
            transcription_result = t.transcribe(audio_path)
            progress.update(
                whisper_task,
                completed=1, total=1,
                description=f"[green]Transcription done ({len(transcription_result.segments)} segments)",
            )

        def _detect_speakers():
            nonlocal frame_results

            def _cb(n: int) -> None:
                progress.update(speaker_task, completed=n)

            frame_results = _run_speaker_detection(frames, config, progress_callback=_cb)
            progress.update(
                speaker_task,
                description="[green]Speaker detection done",
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_transcribe), pool.submit(_detect_speakers)]
            for f in as_completed(futures):
                f.result()  # re-raises any exception

        # ----------------------------------------------------------------
        # Stage 3: Build speaker timeline
        # ----------------------------------------------------------------
        timeline_task = progress.add_task("[cyan]Building speaker timeline…", total=None)
        timeline = build_timeline(
            frame_results,
            frame_interval=config.frame_interval,
            min_span_duration=config.min_span_duration,
            merge_gap=config.timeline_merge_gap,
        )
        progress.update(timeline_task, completed=1, total=1,
                        description=f"[green]Timeline: {len(timeline)} spans")

        # ----------------------------------------------------------------
        # Stage 4: Align transcript to speakers
        # ----------------------------------------------------------------
        align_task = progress.add_task("[cyan]Aligning transcript to speakers…", total=None)
        utterances = align(
            transcription_result,
            timeline,
            merge_gap=config.alignment_merge_gap,
        )
        progress.update(align_task, completed=1, total=1,
                        description=f"[green]Aligned: {len(utterances)} utterances")

        # ----------------------------------------------------------------
        # Stage 5: Generate summary
        # ----------------------------------------------------------------
        summary_task = progress.add_task("[cyan]Generating summary (Ollama)…", total=None)
        gen = SummaryGenerator(
            model=config.ollama_model,
            temperature=config.ollama_temperature,
        )
        summary_md = gen.generate(utterances)
        progress.update(summary_task, completed=1, total=1,
                        description="[green]Summary generated")

        # ----------------------------------------------------------------
        # Stage 6: Write outputs
        # ----------------------------------------------------------------
        write_task = progress.add_task("[cyan]Writing output files…", total=None)
        t_path, s_path = write_outputs(
            utterances, summary_md,
            output_dir=config.output_dir,
            stem=stem,
            video_path=video_path,
        )
        progress.update(write_task, completed=1, total=1,
                        description="[green]Output files written")

    # Collect participant names
    participants: list[str] = sorted(
        {name for span in timeline for name in span.speakers if name != "Unknown"}
    )

    _cached_result = PipelineResult(
        utterances=utterances,
        timeline=timeline,
        transcription=transcription_result,
        transcript_path=t_path,
        summary_path=s_path,
        participants=participants,
    )
