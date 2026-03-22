"""Main processing pipeline with parallel execution.

Architecture (parallel stages):

    ┌── extract_audio ──────────────────────► Whisper (Metal/MLX) ──┐
    │                                                                 ├──► align ──► summarise ──► write
    └── extract_frames ──► speaker_detection (two-pass) ─────────────┘

Parallelism strategy
--------------------
Stage 1 — Extraction:
  Audio and frame extraction run in two threads (both are ffmpeg subprocesses).

Stage 2 — Transcription + Speaker detection (concurrent):
  Whisper and speaker detection run in two threads.  Both are compute-bound
  but use different hardware (Whisper → Metal via MLX; speaker detection →
  MPS for EasyOCR + NEON via OpenCV).  Concurrent execution utilises both.

Speaker detection — two-pass design:
  Pass A (fast, I/O+CPU — frame prefetch thread):
    A background thread reads JPEG frames from disk into a bounded queue while
    the main detection thread processes them.  This hides disk I/O latency.
    Per-frame work: green-blob detection + LayoutTracker update (both fast).

  Pass B (OCR — triggered lazily, amortised):
    EasyOCR is called only when a blob position is not yet in the name cache.
    For a 1-hour meeting (3600 frames, 10 participants) this means ≤10 OCR
    calls total — the rest are O(1) cache lookups.

Stage 3 onwards — sequential (alignment, summary, write) — not a bottleneck.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Callable

import cv2
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
from meeting_transcriber.logging_config import console
from meeting_transcriber.video.extractor import VideoExtractor

log = logging.getLogger(__name__)

# Frames to pre-load into the queue ahead of processing.
# Larger → more RAM; smaller → more stalls.  8 is a good trade-off.
_PREFETCH_DEPTH = 8


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Tunable parameters for the processing pipeline."""

    # Whisper
    # mlx-whisper uses temperature-based sampling; beam search is not supported.
    # temperature=0.0 → greedy decoding (deterministic, fastest, high quality).
    whisper_model: str = "large-v3"
    whisper_temperature: float = 0.0

    # Frame sampling — 0.5 s gives better temporal resolution for the
    # intermittent WebEx green cue without significantly increasing total cost
    # (speaker detection is fast compared to Whisper transcription).
    frame_interval: float = 0.5

    # Speaker detection
    layout_window: int = 10           # frames for layout accumulation
    layout_drift_threshold: int = 80  # px drift before re-calibration
    mic_cluster_radius: int = 40      # px radius for blob deduplication

    # Timeline
    # WebEx visual cues (green mic, tile border, name highlight) can lag actual
    # speech onset by up to ~1 s and drop intermittently during speech.
    # Use a generous merge gap and padding to bridge those gaps.
    timeline_merge_gap: float = 3.0   # seconds; merge same-speaker gaps
    min_span_duration: float = 0.5    # drop spans shorter than this
    timeline_onset_pad: float = 1.0   # expand span starts back by ~1 s
    timeline_offset_pad: float = 0.5  # expand span ends forward by ~0.5 s
    # Temporal smoothing: majority-vote window over N consecutive frames
    timeline_smoothing_window: int = 3
    # Alignment boundary tolerance: widens per-segment speaker lookup (seconds)
    alignment_boundary_tolerance: float = 1.5

    # Alignment
    alignment_merge_gap: float = 1.5  # seconds; merge same-speaker segments

    # Summary
    ollama_model: str = "gemma3:1b"
    ollama_temperature: float = 0.2

    # Output
    output_dir: Path = field(default_factory=lambda: Path("."))
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
# Frame prefetch helper
# ---------------------------------------------------------------------------

_SENTINEL = object()  # signals end-of-stream to the consumer


def _prefetch_frames(
    frames: list[tuple[float, Path]],
    queue: Queue,
) -> None:
    """Background thread: read JPEG frames from disk into a bounded queue."""
    for ts, path in frames:
        frame = cv2.imread(str(path))
        if frame is None:
            log.warning("Unreadable frame at t=%.1fs: %s", ts, path.name)
            continue
        queue.put((ts, frame))
    queue.put(_SENTINEL)


# ---------------------------------------------------------------------------
# Speaker detection (two-pass with prefetch)
# ---------------------------------------------------------------------------

def _run_speaker_detection(
    frames: list[tuple[float, Path]],
    config: PipelineConfig,
    progress_callback: Callable[[int], None] | None = None,
) -> list[tuple[float, list]]:
    """Detect active speakers across all frames.

    Uses a background prefetch thread to hide disk I/O latency.
    EasyOCR is called lazily and only for previously unseen tile positions;
    subsequent lookups are O(1) cache reads.
    """
    log.info("Speaker detection starting: %d frames to process", len(frames))

    tracker = LayoutTracker(
        window=config.layout_window,
        drift_threshold=config.layout_drift_threshold,
    )
    detector = SpeakerDetector(mic_cluster_radius=config.mic_cluster_radius)

    # OCR window: restrict new participant-name OCR to 5%–95% of the video to
    # avoid the dynamic situations at meeting start (join animations, layout
    # changes) and end (leave notifications).  Cached names are still used
    # outside this window; only new OCR calls are suppressed.
    if len(frames) >= 2:
        first_ts, last_ts = frames[0][0], frames[-1][0]
        span = last_ts - first_ts
        ocr_start = first_ts + span * 0.05
        ocr_end   = first_ts + span * 0.95
    else:
        # Too few frames to apply the window; allow OCR everywhere.
        ocr_start, ocr_end = 0.0, float("inf")

    log.info(
        "OCR window: %.1fs – %.1fs (5%%–95%% of video)",
        ocr_start, ocr_end,
    )

    # Start prefetch thread
    queue: Queue = Queue(maxsize=_PREFETCH_DEPTH)
    prefetch_thread = Thread(
        target=_prefetch_frames, args=(frames, queue), daemon=True, name="frame-prefetch"
    )
    prefetch_thread.start()

    results: list[tuple[float, list]] = []
    processed = 0

    while True:
        item = queue.get()
        if item is _SENTINEL:
            break

        ts, frame = item
        layout = tracker.update(ts, frame)
        layout_tiles = layout.tiles if layout else None
        allow_ocr = ocr_start <= ts <= ocr_end
        active = detector.process_frame(frame, layout_tiles=layout_tiles, allow_new_ocr=allow_ocr)
        results.append((ts, active))
        processed += 1

        if active:
            log.debug("t=%.1fs: active = %s", ts, [s.name for s in active])

        if progress_callback:
            progress_callback(processed)

    prefetch_thread.join(timeout=5)
    if prefetch_thread.is_alive():
        log.warning(
            "Frame-prefetch thread did not exit within 5s — "
            "possible I/O stall on frame files (thread is daemon, will be cleaned up on exit)"
        )

    log.info(
        "Speaker detection done: %d frames, %d participant(s) identified: %s",
        len(results),
        len(detector.known_participants),
        detector.known_participants or ["none detected"],
    )
    return results


# ---------------------------------------------------------------------------
# Main pipeline (class-based, no module-level globals)
# ---------------------------------------------------------------------------

class Pipeline:
    """Orchestrates the full meeting transcription pipeline."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(self, video_path: Path) -> PipelineResult:
        """Run the full pipeline on a video file.

        Args:
            video_path: Path to the input MP4.

        Returns:
            PipelineResult containing paths to generated Markdown files
            and structured data.
        """
        video_path = Path(video_path)
        stem = video_path.stem
        cfg = self.config

        work_dir = Path(tempfile.mkdtemp(prefix="meeting_work_"))
        log.info("Work directory: %s", work_dir)
        log.info(
            "Pipeline config: whisper=%s, ollama=%s, frame_interval=%.1fs",
            cfg.whisper_model, cfg.ollama_model, cfg.frame_interval,
        )

        try:
            result = self._run(video_path, stem, work_dir)
        finally:
            if not cfg.keep_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

        return result

    def _run(self, video_path: Path, stem: str, work_dir: Path) -> PipelineResult:
        cfg = self.config

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
            extractor = VideoExtractor(video_path, frame_interval=cfg.frame_interval)
            audio_path, frames = extractor.extract_all(work_dir)
            total_frames = len(frames)
            progress.update(
                extract_task, completed=1, total=1,
                description=f"[green]Extracted {total_frames} frames + audio",
            )

            # ----------------------------------------------------------------
            # Stage 2: Whisper + speaker detection concurrently
            # ----------------------------------------------------------------
            whisper_task = progress.add_task("[cyan]Transcribing audio (Whisper)…", total=None)
            speaker_task = progress.add_task(
                "[cyan]Detecting speakers…", total=total_frames
            )

            transcription_result: TranscriptionResult | None = None
            frame_results: list[tuple[float, list]] | None = None

            def _transcribe() -> None:
                nonlocal transcription_result
                t = WhisperTranscriber(
                    model=cfg.whisper_model,
                    temperature=cfg.whisper_temperature,
                )
                transcription_result = t.transcribe(audio_path)
                progress.update(
                    whisper_task,
                    completed=1, total=1,
                    description=(
                        f"[green]Transcription done "
                        f"({len(transcription_result.segments)} segments)"
                    ),
                )

            def _detect_speakers() -> None:
                nonlocal frame_results

                def _cb(n: int) -> None:
                    progress.update(speaker_task, completed=n)

                frame_results = _run_speaker_detection(frames, cfg, progress_callback=_cb)
                progress.update(
                    speaker_task,
                    description="[green]Speaker detection done",
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(_transcribe), pool.submit(_detect_speakers)]
                for f in as_completed(futures):
                    f.result()  # re-raises any exception from the thread

            # ----------------------------------------------------------------
            # Stage 3: Build speaker timeline
            # ----------------------------------------------------------------
            timeline_task = progress.add_task(
                "[cyan]Building speaker timeline…", total=None
            )
            timeline = build_timeline(
                frame_results,
                frame_interval=cfg.frame_interval,
                min_span_duration=cfg.min_span_duration,
                merge_gap=cfg.timeline_merge_gap,
                onset_pad=cfg.timeline_onset_pad,
                offset_pad=cfg.timeline_offset_pad,
                smoothing_window=cfg.timeline_smoothing_window,
            )
            progress.update(
                timeline_task, completed=1, total=1,
                description=f"[green]Timeline: {len(timeline)} spans",
            )

            # ----------------------------------------------------------------
            # Stage 4: Align transcript to speakers
            # ----------------------------------------------------------------
            align_task = progress.add_task(
                "[cyan]Aligning transcript to speakers…", total=None
            )
            utterances = align(
                transcription_result,
                timeline,
                merge_gap=cfg.alignment_merge_gap,
                boundary_tolerance=cfg.alignment_boundary_tolerance,
            )
            progress.update(
                align_task, completed=1, total=1,
                description=f"[green]Aligned: {len(utterances)} utterances",
            )

            # ----------------------------------------------------------------
            # Stage 5: Generate summary
            # ----------------------------------------------------------------
            summary_task = progress.add_task(
                "[cyan]Generating summary (Ollama)…", total=None
            )
            gen = SummaryGenerator(
                model=cfg.ollama_model,
                temperature=cfg.ollama_temperature,
            )
            summary_md = gen.generate(utterances)
            progress.update(
                summary_task, completed=1, total=1,
                description="[green]Summary generated",
            )

            # ----------------------------------------------------------------
            # Stage 6: Write outputs
            # ----------------------------------------------------------------
            write_task = progress.add_task("[cyan]Writing output files…", total=None)
            t_path, s_path = write_outputs(
                utterances, summary_md,
                output_dir=cfg.output_dir,
                stem=stem,
                video_path=video_path,
            )
            progress.update(
                write_task, completed=1, total=1,
                description="[green]Output files written",
            )

        participants = sorted(
            {name for span in timeline for name in span.speakers if name != "Unknown"}
        )

        return PipelineResult(
            utterances=utterances,
            timeline=timeline,
            transcription=transcription_result,
            transcript_path=t_path,
            summary_path=s_path,
            participants=participants,
        )


# ---------------------------------------------------------------------------
# Convenience function (backwards-compatible with original run() signature)
# ---------------------------------------------------------------------------

def run(
    video_path: Path,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the full meeting transcription pipeline (convenience wrapper)."""
    return Pipeline(config).run(video_path)
