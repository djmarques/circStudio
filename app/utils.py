"""Shared helpers for the circStudio Shiny app.

This module centralises file-format dispatching and the batch-loading data
structures (``BatchEntry`` / ``BatchCollection``) used by the upload and
batch-overview tabs. Every analysis tab also relies on the small ``active_raw``
helper to resolve which ``Raw`` object is currently selected.

The circStudio package is treated as a read-only dependency; nothing here
modifies anything under ``src/``.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Directories that should never be treated as a batch (avoids accidentally
# scanning an entire project tree if the user points Browse at e.g. a repo).
_SKIP_DIR_NAMES = {
    "__pycache__", "site-packages", "node_modules", ".venv", "venv",
    ".git", ".idea", ".pytest_cache", "dist", "build", ".ipynb_checkpoints",
}


def _skippable(rel_parts) -> bool:
    """True if any path component is hidden or a known system directory."""
    return any(
        p.startswith(".") or p in _SKIP_DIR_NAMES for p in rel_parts
    )

# circStudio readers -------------------------------------------------------
from circstudio.io import (
    read_atr,
    read_awd,
    read_agd,
    read_dqt,
    read_mesa,
    read_rpx,
    read_tal,
)

# Supported file extensions for both single-file and batch loading.
SUPPORTED_EXTENSIONS = {".awd", ".agd", ".csv", ".mtn", ".txt"}

# Human-friendly format names used by the explicit format override in the UI.
READERS = {
    "awd": read_awd,
    "agd": read_agd,
    "rpx": read_rpx,
    "dqt": read_dqt,
    "mesa": read_mesa,
    "tal": read_tal,
    "atr": read_atr,
}

# Default reader to try first for a given extension. Some extensions (.csv,
# .txt) are shared by several formats, so ``dispatch_reader`` falls back through
# an ordered list of candidates when the first attempt fails.
_EXT_CANDIDATES = {
    ".awd": ["awd"],
    ".agd": ["agd"],
    ".csv": ["rpx", "dqt", "mesa"],
    ".mtn": ["tal", "atr"],
    ".txt": ["atr", "tal"],
}


# The ActTrust (Condor Instruments) report banner that marks the true start of
# an ATR header, e.g. "+----+ Condor Instruments Report +----+".
_ATR_BANNER = re.compile(r"\+-*\+ \w+ \w+ \w+ \+-*\+")


def detect_atr_skiprows(filepath: Path) -> int:
    """Count lines that appear *above* the Condor Instruments Report banner.

    ActTrust exports sometimes carry extra preamble lines (for example
    ``#ActLogModel=2.0.0``) before the ``+----+ Condor Instruments Report
    +----+`` banner. The ATR reader expects that banner to be the first line,
    so the number returned here is passed as ``skip_rows``.
    """
    try:
        with open(filepath, errors="ignore") as fp:
            for i, line in enumerate(fp):
                if _ATR_BANNER.match(line.strip()) or _ATR_BANNER.match(line):
                    return i
                if i > 50:  # banner should appear early; stop scanning otherwise
                    break
    except Exception:
        pass
    return 0


def _reader_kwargs(key: str, filepath: Path) -> dict:
    """Per-format keyword arguments computed from the file contents."""
    if key == "atr":
        return {"skip_rows": detect_atr_skiprows(filepath)}
    return {}


# --------------------------------------------------------------------------
# Single-file dispatching
# --------------------------------------------------------------------------
def dispatch_reader(filepath: Path, fmt: Optional[str] = None) -> object:
    """Map a file to the correct ``read_*`` function and call it.

    Parameters
    ----------
    filepath : Path
        Path to the actigraphy file.
    fmt : str, optional
        Explicit format key (one of ``READERS``). When given, only that reader
        is used. When ``None`` the extension decides the candidate readers and
        the first one that succeeds wins.

    Returns
    -------
    object
        A circStudio ``Raw`` instance.

    Raises
    ------
    ValueError
        If no candidate reader can parse the file.
    """
    filepath = Path(filepath)

    if fmt is not None:
        fmt = fmt.lower()
        if fmt not in READERS:
            raise ValueError(
                f"Unknown format '{fmt}'. Available: {sorted(READERS)}"
            )
        return READERS[fmt](str(filepath), **_reader_kwargs(fmt, filepath))

    ext = filepath.suffix.lower()
    candidates = _EXT_CANDIDATES.get(ext)
    if not candidates:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # For .txt files the ATR and TAL readers share the extension. If the file
    # carries the ActTrust/Condor banner it is unambiguously ATR — read it with
    # ONLY the ATR reader and let any ATR error propagate immediately, rather
    # than falling through to the TAL reader (which hangs on ATR-format input).
    if ext == ".txt" and _ATR_BANNER.search(
        open(filepath, errors="ignore").read(512)
    ):
        key = "atr"
        return READERS[key](str(filepath), **_reader_kwargs(key, filepath))

    errors = []
    for key in candidates:
        try:
            return READERS[key](str(filepath), **_reader_kwargs(key, filepath))
        except Exception as exc:  # pragma: no cover - depends on file content
            errors.append(f"{key}: {exc}")

    raise ValueError(
        "Could not read '{}'. Tried: {}.".format(
            filepath.name, "; ".join(errors)
        )
    )


# --------------------------------------------------------------------------
# Raw-object summary helpers
# --------------------------------------------------------------------------
def raw_has_light(raw) -> bool:
    """Return True if the Raw object carries a usable light channel."""
    light = getattr(raw, "light", None)
    try:
        return light is not None and len(light) > 0
    except TypeError:
        return False


def raw_sampling_freq(raw) -> str:
    """Best-effort string for the sampling frequency of a Raw object."""
    freq = getattr(raw, "frequency", None)
    if freq is not None:
        return str(freq)
    try:
        return str(raw.activity.index.freq)
    except Exception:
        return "unknown"


def raw_summary_row(raw) -> dict:
    """Build a one-row metadata dict for a single Raw object."""
    act = raw.activity
    try:
        duration = str(raw.duration())
    except Exception:
        duration = str(act.index[-1] - act.index[0])
    return {
        "n_epochs": len(act),
        "sampling_freq": raw_sampling_freq(raw),
        "duration": duration,
        "start_time": str(act.index[0]),
        "end_time": str(act.index[-1]),
        "has_light": "yes" if raw_has_light(raw) else "no",
    }


# --------------------------------------------------------------------------
# Batch loading data structures
# --------------------------------------------------------------------------
@dataclass
class BatchEntry:
    """A single successfully loaded recording inside a batch."""

    subject_id: str
    factor_levels: tuple[str, ...]  # e.g. ("male",) or ("pre", "male")
    filepath: Path
    raw: object  # circStudio Raw instance


@dataclass
class BatchCollection:
    """A collection of recordings loaded from a directory tree."""

    entries: list[BatchEntry] = field(default_factory=list)
    factor_names: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # -- introspection -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def subject_ids(self) -> list[str]:
        return [e.subject_id for e in self.entries]

    def levels_for_factor(self, factor_index: int = 0) -> list[str]:
        """Sorted unique levels for the requested factor (0-based)."""
        levels = {
            e.factor_levels[factor_index]
            for e in self.entries
            if len(e.factor_levels) > factor_index
        }
        return sorted(levels)

    def filter(self, factor_index: int = 0, level: Optional[str] = None):
        """Return entries whose ``factor_index`` level equals ``level``."""
        if level is None:
            return list(self.entries)
        return [
            e
            for e in self.entries
            if len(e.factor_levels) > factor_index
            and e.factor_levels[factor_index] == level
        ]

    # -- lookup ------------------------------------------------------------
    def get(self, subject_id: str, *factor_levels: str) -> Optional[BatchEntry]:
        """Look up an entry by subject id (and optionally factor levels)."""
        for e in self.entries:
            if e.subject_id != subject_id:
                continue
            if factor_levels and tuple(factor_levels) != e.factor_levels:
                continue
            return e
        return None

    # -- tabular view ------------------------------------------------------
    def to_metadata_df(self) -> pd.DataFrame:
        """Metadata table for every loaded recording.

        Columns: subject_id, factor_1[, factor_2, ...], duration,
        sampling_freq, start_time, has_light.
        """
        rows = []
        for e in self.entries:
            row = {"subject_id": e.subject_id}
            for i, name in enumerate(self.factor_names):
                col = f"factor_{i + 1}"
                level = e.factor_levels[i] if i < len(e.factor_levels) else ""
                row[col] = level
            summary = raw_summary_row(e.raw)
            row["duration"] = summary["duration"]
            row["sampling_freq"] = summary["sampling_freq"]
            row["start_time"] = summary["start_time"]
            row["has_light"] = summary["has_light"]
            rows.append(row)
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Directory scanning
# --------------------------------------------------------------------------
def _detect_depth(root: Path) -> int:
    """Detect the factor depth of a batch directory.

    Returns 1 for a flat (one-factor) layout where the immediate
    subdirectories contain data files directly, or 2 for a nested
    (two-factor) layout where the immediate subdirectories only contain
    further subdirectories.
    """
    subdirs = [
        p for p in root.iterdir()
        if p.is_dir() and not _skippable(p.relative_to(root).parts)
    ]
    if not subdirs:
        return 1

    has_files_at_depth1 = False
    has_subdirs_at_depth1 = False
    for d in subdirs:
        for child in d.iterdir():
            if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                has_files_at_depth1 = True
            elif child.is_dir() and not child.name.startswith("."):
                has_subdirs_at_depth1 = True

    # If files appear directly inside the first-level folders, this is the
    # factor level (depth 1) even if some stray sub-subdirectories exist.
    if has_files_at_depth1:
        return 1
    if has_subdirs_at_depth1:
        return 2
    return 1


def scan_batch_directory(
    root: Path, fmt: Optional[str] = None, per_file_timeout: float = 45.0
) -> BatchCollection:
    """Walk ``root``, detect depth, load every supported file.

    Files that fail to load (or that exceed ``per_file_timeout`` seconds) are
    skipped and recorded in ``collection.errors`` — a single problem file never
    hangs or aborts the whole batch. Hidden/system directories are ignored.

    Parameters
    ----------
    root : Path
        Root directory of the batch.
    fmt : str, optional
        Force a specific reader for every file. When ``None`` the file
        extension drives auto-detection.
    per_file_timeout : float
        Maximum seconds to spend reading any single file before skipping it.
    """
    root = Path(root)
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"'{root}' is not a directory.")

    depth = _detect_depth(root)
    collection = BatchCollection()

    warned_too_deep = False
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        rel_parts = path.relative_to(root).parts[:-1]  # drop the file name

        # Ignore hidden/system directories (e.g. .venv, site-packages, .git).
        if _skippable(rel_parts):
            continue

        # Enforce the detected depth. Deeper files are ignored with a warning.
        if len(rel_parts) > depth:
            if not warned_too_deep:
                collection.errors.append(
                    "Some files are nested deeper than the detected factor "
                    f"depth ({depth}); those sub-levels were ignored."
                )
                warned_too_deep = True
            rel_parts = rel_parts[:depth]

        factor_levels = tuple(rel_parts)
        subject_id = path.stem

        # Read with a per-file timeout so one slow/hanging file cannot stall
        # the entire batch. A daemon thread is abandoned on timeout (it will
        # not block interpreter/app shutdown).
        box: dict = {}

        def _worker(p=path):
            try:
                box["raw"] = dispatch_reader(p, fmt)
            except Exception as exc:  # noqa: BLE001
                box["err"] = exc

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        th.join(per_file_timeout)
        if th.is_alive():
            collection.errors.append(
                f"{path.name}: timed out after {per_file_timeout:.0f}s "
                "(try selecting the exact File format instead of Auto-detect)."
            )
            continue
        if "err" in box:
            collection.errors.append(f"{path.name}: {box['err']}")
            continue
        raw = box["raw"]

        collection.entries.append(
            BatchEntry(
                subject_id=subject_id,
                factor_levels=factor_levels,
                filepath=path,
                raw=raw,
            )
        )

    # Infer factor names from the observed depth.
    if depth == 1:
        collection.factor_names = ["factor_1"]
    elif depth == 2:
        collection.factor_names = ["factor_1", "factor_2"]
    else:
        collection.factor_names = [f"factor_{i + 1}" for i in range(depth)]

    # If nothing actually had factor levels (files directly under root), drop
    # the placeholder factor name.
    if collection.entries and all(
        len(e.factor_levels) == 0 for e in collection.entries
    ):
        collection.factor_names = []

    return collection


# --------------------------------------------------------------------------
# Active-series resolution shared by all analysis tabs
# --------------------------------------------------------------------------
def active_raw(rv_mode, rv_single, rv_batch, subject_id: Optional[str] = None):
    """Resolve the Raw object an analysis tab should operate on.

    In single mode this is simply ``rv_single``. In batch mode it is the entry
    matching ``subject_id`` (falling back to the first entry).
    Returns ``None`` when nothing is available.
    """
    mode = rv_mode()
    if mode == "single":
        return rv_single()

    batch = rv_batch()
    if batch is None or len(batch) == 0:
        return None
    if subject_id:
        entry = batch.get(subject_id)
        if entry is not None:
            return entry.raw
    return batch.entries[0].raw
