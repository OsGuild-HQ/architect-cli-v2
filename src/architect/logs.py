"""File logging for the `architect` logger.

Nothing is ever written to stdout/stderr: a Textual app owns the terminal, so
stray log output would corrupt the screen.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("architect")
log.propagate = False
# Without a handler Python falls back to `logging.lastResort`, which prints
# warnings to stderr. A NullHandler keeps the logger silent until set up.
log.addHandler(logging.NullHandler())


def setup_logging(path: Path, level: int = logging.INFO) -> Path:
    """Attach a file handler once. Safe to call repeatedly."""
    if any(isinstance(h, logging.FileHandler) for h in log.handlers):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    log.addHandler(handler)
    log.setLevel(level)
    return path
