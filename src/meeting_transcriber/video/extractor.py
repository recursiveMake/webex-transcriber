"""Extract audio and frames from an MP4 file using ffmpeg."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class VideoExtractor:
    """Extracts audio and sampled frames from a video file."""

    def __init__(self, video_path: Path, frame_interval: float = 1.0) -> None:
        self.video_path = Path(video_path)
        self.frame_interval = frame_interval
        self._probe_data: dict | None = None

    # ------------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------------

    @property
    def probe(self) -> dict:
        if self._probe_data is None:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_format", "-show_streams",
                    str(self.video_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            import json
            self._probe_data = json.loads(result.stdout)
        return self._probe_data

    @property
    def duration(self) -> float:
        return float(self.probe["format"]["duration"])

    @property
    def fps(self) -> float:
        video_stream = next(
            s for s in self.probe["streams"] if s["codec_type"] == "video"
        )
        num, den = video_stream["r_frame_rate"].split("/")
        return float(num) / float(den)

    @property
    def resolution(self) -> tuple[int, int]:
        """Return (width, height)."""
        video_stream = next(
            s for s in self.probe["streams"] if s["codec_type"] == "video"
        )
        return int(video_stream["width"]), int(video_stream["height"])

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_audio(self, output_path: Path) -> Path:
        """Extract mono 16 kHz WAV suitable for Whisper."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(self.video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(output_path),
            ],
            capture_output=True,
            check=True,
        )
        return output_path

    def extract_frames(self, output_dir: Path) -> list[tuple[float, Path]]:
        """Extract one frame every `frame_interval` seconds as JPEG.

        Returns a sorted list of (timestamp_seconds, path) pairs.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fps_filter = f"1/{self.frame_interval}" if self.frame_interval != 1.0 else "1"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(self.video_path),
                "-vf", f"fps={fps_filter}",
                "-q:v", "2",
                str(output_dir / "frame_%06d.jpg"),
            ],
            capture_output=True,
            check=True,
        )
        frames = sorted(output_dir.glob("frame_*.jpg"))
        return [(i * self.frame_interval, f) for i, f in enumerate(frames)]

    def extract_all(
        self, work_dir: Path
    ) -> tuple[Path, list[tuple[float, Path]]]:
        """Extract audio and frames in parallel. Returns (audio_path, frames)."""
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        audio_path = work_dir / "audio.wav"
        frames_dir = work_dir / "frames"

        with ThreadPoolExecutor(max_workers=2) as pool:
            audio_future = pool.submit(self.extract_audio, audio_path)
            frames_future = pool.submit(self.extract_frames, frames_dir)
            return audio_future.result(), frames_future.result()
