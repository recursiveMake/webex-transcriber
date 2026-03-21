"""Logging configuration for meeting-transcriber.

Sets up a single root logger for the ``meeting_transcriber`` package using
Python's standard ``logging`` module.  Rich is used for console output only
in the CLI/pipeline; library code emits plain log records.

Log levels:
  DEBUG   — per-frame detail, cache hits/misses, blob counts
  INFO    — stage start/end, counts, model choices  (shown by default)
  WARNING — recoverable anomalies (OCR failure, missing speaker, etc.)
  ERROR   — unrecoverable failures
"""

from __future__ import annotations

import logging
import sys


_PACKAGE = "meeting_transcriber"
_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FMT = "%H:%M:%S"


def setup(level: int = logging.INFO, *, quiet: bool = False) -> None:
    """Configure package-level logging.

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

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(effective)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    pkg_logger.addHandler(handler)

    # Suppress overly chatty third-party loggers that leak through
    for noisy in ("easyocr", "PIL", "torch", "mlx", "filelock", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
