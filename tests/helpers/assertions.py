"""Reusable, domain-specific assertions for the circStudio test suite."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def assert_datetime_index_regular(obj, name: str = "series") -> None:
    """The object is indexed by a strictly increasing, evenly spaced DatetimeIndex."""
    index = getattr(obj, "index", None)
    assert isinstance(index, pd.DatetimeIndex), (
        f"{name}: expected a DatetimeIndex, got {type(index).__name__}"
    )
    assert index.is_monotonic_increasing, f"{name}: DatetimeIndex is not monotonic increasing"
    assert index.is_unique, f"{name}: DatetimeIndex contains duplicate timestamps"
    if len(index) > 2:
        deltas = np.diff(index.values)
        assert len(set(deltas.tolist())) == 1, (
            f"{name}: sampling is irregular; found {len(set(deltas.tolist()))} distinct gaps"
        )


def assert_valid_activity_series(obj, name: str = "activity", allow_nan: bool = False) -> None:
    """The object is a numeric, regularly-sampled, non-empty activity Series."""
    assert isinstance(obj, pd.Series), f"{name}: expected pd.Series, got {type(obj).__name__}"
    assert len(obj) > 0, f"{name}: series is empty"
    assert pd.api.types.is_numeric_dtype(obj), f"{name}: dtype {obj.dtype} is not numeric"
    assert_datetime_index_regular(obj, name=name)
    if not allow_nan:
        n_nan = int(obj.isna().sum())
        assert n_nan == 0, f"{name}: contains {n_nan} NaN values"


def assert_within_range(
    value, low: float, high: float, name: str = "value", inclusive: bool = True
) -> None:
    """Scalar or array lies within ``[low, high]`` (or the open interval)."""
    arr = np.asarray(value, dtype=float)
    finite = arr[np.isfinite(arr)]
    assert finite.size > 0, f"{name}: no finite values to range-check"
    if inclusive:
        ok = (finite >= low) & (finite <= high)
    else:
        ok = (finite > low) & (finite < high)
    assert ok.all(), (
        f"{name}: {int((~ok).sum())} of {finite.size} values outside "
        f"[{low}, {high}] (min={finite.min()!r}, max={finite.max()!r})"
    )


def assert_no_nan_unless_expected(obj, expected: int = 0, name: str = "result") -> None:
    """Exactly ``expected`` NaN values are present."""
    arr = np.asarray(obj, dtype=float)
    n_nan = int(np.isnan(arr).sum())
    assert n_nan == expected, f"{name}: expected {expected} NaN, found {n_nan}"


def assert_bouts_partition_recording(
    sleep_bouts, active_bouts, total_epochs: int, name: str = "bouts", tolerance: int = 0
) -> None:
    """Sleep and active bouts together cover the recording exactly once."""

    def _count(bouts) -> int:
        total = 0
        for bout in bouts:
            if hasattr(bout, "__len__") and len(bout) == 2 and not isinstance(bout, str):
                start, end = bout
                total += int(end) - int(start)
            elif hasattr(bout, "duration"):
                total += int(bout.duration)
            else:
                total += len(bout)
        return total

    covered = _count(sleep_bouts) + _count(active_bouts)
    assert abs(covered - total_epochs) <= tolerance, (
        f"{name}: sleep+active bouts cover {covered} epochs but the recording "
        f"has {total_epochs} (tolerance {tolerance})"
    )


def assert_figure_renders(fig, min_traces: int = 1, name: str = "figure") -> None:
    """A matplotlib or plotly figure was produced and carries data."""
    assert fig is not None, f"{name}: no figure returned"

    # Plotly
    if hasattr(fig, "data") and hasattr(fig, "layout"):
        assert len(fig.data) >= min_traces, (
            f"{name}: plotly figure has {len(fig.data)} traces, expected >= {min_traces}"
        )
        return

    # Matplotlib Figure
    if hasattr(fig, "axes"):
        assert len(fig.axes) >= 1, f"{name}: matplotlib figure has no axes"
        n_artists = sum(len(ax.lines) + len(ax.patches) + len(ax.collections) for ax in fig.axes)
        assert n_artists >= min_traces, (
            f"{name}: matplotlib figure has {n_artists} artists, expected >= {min_traces}"
        )
        return

    # Matplotlib Axes
    if hasattr(fig, "get_figure"):
        n_artists = len(fig.lines) + len(fig.patches) + len(fig.collections)
        assert n_artists >= min_traces, (
            f"{name}: axes has {n_artists} artists, expected >= {min_traces}"
        )
        return

    raise AssertionError(f"{name}: unrecognised figure type {type(fig).__name__}")


def assert_monotonic(values, increasing: bool = True, strict: bool = False, name: str = "sequence") -> None:
    """The sequence is (strictly) monotonic in the requested direction."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    diffs = np.diff(arr)
    if increasing:
        ok = (diffs > 0).all() if strict else (diffs >= 0).all()
        direction = "strictly increasing" if strict else "non-decreasing"
    else:
        ok = (diffs < 0).all() if strict else (diffs <= 0).all()
        direction = "strictly decreasing" if strict else "non-increasing"
    assert ok, f"{name}: not {direction}; violations at {np.flatnonzero(~(diffs > 0 if strict and increasing else diffs >= 0)).tolist()[:5]}"


def assert_close(actual, expected, rtol: float = 1e-7, atol: float = 0.0, name: str = "value") -> None:
    """Thin wrapper over ``np.allclose`` with a message naming the quantity."""
    actual_arr = np.asarray(actual, dtype=float)
    expected_arr = np.asarray(expected, dtype=float)
    assert np.allclose(actual_arr, expected_arr, rtol=rtol, atol=atol, equal_nan=True), (
        f"{name}: {actual!r} != {expected!r} (rtol={rtol}, atol={atol})"
    )


def assert_timedelta_close(actual, expected, tolerance: pd.Timedelta, name: str = "duration") -> None:
    """Two Timedelta-like quantities agree within ``tolerance``."""
    actual_td = pd.Timedelta(actual)
    expected_td = pd.Timedelta(expected)
    delta = abs(actual_td - expected_td)
    assert delta <= tolerance, (
        f"{name}: {actual_td} differs from {expected_td} by {delta} (tolerance {tolerance})"
    )


def count_runs(mask) -> list[tuple[int, int]]:
    """Return ``(start, length)`` for every run of True in a boolean sequence."""
    arr = np.asarray(mask).astype(bool)
    if arr.size == 0:
        return []
    padded = np.concatenate([[False], arr, [False]])
    diffs = np.diff(padded.astype(int))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def assert_runs_equal(mask, expected_runs, name: str = "mask") -> None:
    """The boolean mask contains exactly the expected ``(start, length)`` runs."""
    actual = count_runs(mask)
    expected = [(int(s), int(length)) for s, length in expected_runs]
    assert actual == expected, f"{name}: runs {actual} != expected {expected}"


def assert_is_binary(obj, name: str = "series", values=(0, 1)) -> None:
    """Every value is one of ``values`` (ignoring NaN)."""
    arr = np.asarray(obj, dtype=float)
    finite = arr[np.isfinite(arr)]
    allowed = set(float(v) for v in values)
    found = set(np.unique(finite).tolist())
    assert found.issubset(allowed), f"{name}: found values {sorted(found - allowed)} outside {sorted(allowed)}"


def hours_to_timedelta(hours: float) -> pd.Timedelta:
    """Convenience: fractional hours -> Timedelta, exact to the microsecond."""
    if not math.isfinite(hours):
        raise ValueError("hours must be finite")
    return pd.Timedelta(seconds=round(hours * 3600, 6))
