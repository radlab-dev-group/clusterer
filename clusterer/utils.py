"""
Utility helpers for logging, file checks, and streaming JSONL parsing.

This module provides:
- get_logger: configured logger factory with a consistent formatter.
- exists_input_file: small convenience to validate input file paths.
- load_jsonl_file_yield: memory‑efficient JSONL reader that yields parsed
  objects line by line, tolerating malformed lines.
"""

import os
import json
import logging

from typing import Any, Generator


def get_logger():
    """
    Create and return a module-scoped logger.

    The logger is configured with a basic configuration:
    - Level: DEBUG
    - Format: "{asctime} - {levelname} - {message}"
    - Date format: "%Y-%m-%d %H:%M"

    Returns
    -------
    logging.Logger, Logger instance bound to this module's namespace.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    return logging.getLogger(__name__)


def exists_input_file(file_path: str):
    """
    Check whether a given path points to an existing regular file.

    Parameters
    ----------
    file_path : str
        Path to the file to validate.

    Returns
    -------
    bool
        True if the path exists and is a file; otherwise False.
    """
    return os.path.exists(file_path) and os.path.isfile(file_path)


def load_jsonl_file_yield(file_path: str) -> Generator[Any | None, Any, None]:
    """
    Iterate over a JSON Lines (JSONL) file yielding parsed records.

    Each line of the file is parsed independently. If a line cannot be
    decoded as JSON, the generator yields None for that line instead of
    raising an exception, allowing callers to handle malformed rows.

    Parameters
    ----------
    file_path : str
        Path to the JSONL file.

    Yields
    ------
    Any | None
        The parsed JSON object for a line, or None if the line is malformed.

    Notes
    -----
    - If the file does not exist, the generator returns immediately (no yield).
    - The function finishes by returning None after the file is fully consumed.
    """
    if not exists_input_file(file_path):
        return None

    with open(file_path, "r") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield None
        return None
