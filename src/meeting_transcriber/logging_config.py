"""Logging configuration for meeting-transcriber.

All output — both log records and Rich progress bars — must go through the
**same** ``rich.console.Console`` instance.  When two writers share the same
file descriptor but are unaware of each other, Rich redraws its progress bars
on top of any text written between frames, producing duplicated/jumped lines.

Using ``RichHandler(console=console)`` routes log records through Rich's live
display engine, so it can interleave them cleanly above the progress bar.

Log levels:
  DEBUG   — per-frame detail, cache hits/misses, blob counts
  INFO    — stage start/end, counts, model choices  (shown by default)
  WARNING — recoverable anomalies (OCR failure, missing speaker, etc.)
  ERROR   — unrecoverable failures
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Shared console — import this wherever Rich output is needed so that the
# progress bars and the log handler always use the same instance.
# ---------------------------------------------------------------------------
console = Console(stderr=True)

_PACKAGE = "meeting_transcriber"


def setup(level: int = logging.INFO, *, quiet: bool = False) -> None:
    """Configure package-level logging via Rich.

    Args:
        level: Root log level for the ``meeting_transcriber`` package.
        quiet: When True, only WARNING and above are shown.
    """
    effective = logging.WARNING if quiet else level

    pkg_logger = logging.getLogger(_PACKAGE)
    pkg_logger.setLevel(effective)

    # Avoid adding duplicate handlers if called more than once
    if pkg_logger.handlers:
        return

    handler = RichHandler(
        console=console,       # same instance as the progress bars
        show_path=False,       # don't print source file:line
        rich_tracebacks=False,
        markup=False,          # don't interpret log strings as Rich markup
        log_time_format="[%H:%M:%S]",
    )
    handler.setLevel(effective)
    pkg_logger.addHandler(handler)

    # Suppress overly chatty third-party loggers that leak through
    for noisy in ("easyocr", "PIL", "torch", "mlx", "filelock", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
