"""Deterministic two-text-file input and safe overwrite of local output.

Only root-level .txt files are consumed. There are exactly two; neither
filenames nor file metadata are interpolated into the research question.
The combined prompt is written as UTF-8 with one blank-line separator.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile


class InputFolderError(ValueError):
    """A folder cannot provide exactly two nonempty UTF-8 text documents."""


def read_two_text_files(folder: Path) -> tuple[str, tuple[Path, Path]]:
    """Read two text files in case-insensitive filename order, with no API use."""
    folder = Path(folder)
    if not folder.is_dir():
        raise InputFolderError(
            f"Missing input folder: {folder}. Create it with exactly two .txt files."
        )
    try:
        paths = sorted(
            (p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() == ".txt"),
            key=lambda p: (p.name.casefold(), p.name),
        )
    except OSError as exc:
        raise InputFolderError(f"Cannot list {folder}: {exc}") from exc
    if len(paths) != 2:
        raise InputFolderError(
            f"Expected exactly two .txt files in {folder}; found {len(paths)}. "
            "Rename or move extra .txt files before running Council."
        )
    parts = []
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise InputFolderError(
                f"Cannot read UTF-8 text file {path}: {exc}"
            ) from exc
        if not content.strip():
            raise InputFolderError(f"Text file is empty: {path}")
        parts.append(content.rstrip("\r\n"))
    combined = "\n\n".join(parts) + "\n"
    return combined, (paths[0], paths[1])


def atomic_write_text(destination: Path, contents: str) -> None:
    """Replace a stable output filename atomically; never append old answers."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        fd, name = tempfile.mkstemp(
            prefix="." + destination.name + ".", suffix=".tmp",
            dir=destination.parent
        )
        temporary = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as writer:
            writer.write(contents)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
