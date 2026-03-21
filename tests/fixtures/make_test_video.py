"""Generate a synthetic WebEx-like test video for integration testing.

Creates a short MP4 (default 12 seconds) with:
- Two participant tiles (Alice and Bob) in a right-side strip
- Alternating active speakers (green mic)
- A sine-wave audio track (no real speech; Whisper produces minimal output)

Usage (from project root):
    .venv/bin/python tests/fixtures/make_test_video.py [output_path] [duration]
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from tests.fixtures.synthetic import two_participant_frame, FRAME_W, FRAME_H


def make_test_video(
    output_path: Path,
    duration: float = 12.0,
    fps: int = 5,
) -> Path:
    """Write a synthetic WebEx MP4 to output_path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = int(duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Write video to a temp file, then mux with audio via ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (FRAME_W, FRAME_H))
    for i in range(total_frames):
        t = i / fps
        # Alice speaks first 6 seconds, Bob the second 6 seconds
        active = "Alice" if t < duration / 2 else "Bob"
        frame = two_participant_frame(active=active)
        writer.write(frame)
    writer.release()

    # Add silent audio via ffmpeg
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(tmp_path),
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )
    tmp_path.unlink(missing_ok=True)
    return output_path


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_meeting.mp4")
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    result = make_test_video(out, duration=dur)
    print(f"Written: {result}")
