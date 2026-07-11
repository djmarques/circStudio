"""Automatic non-wear detection algorithms for actigraphy count data.

Two peer-reviewed algorithms are implemented:

Troiano (2008)
--------------
Troiano RP, Berrigan D, Dodd KW, Mâsse LC, Tilert T, McDowell M.
"Physical activity in the United States measured by accelerometer."
Med Sci Sports Exerc. 2008;40(1):181-188.

The original NHANES non-wear rule: a period is non-wear if it contains
≥ ``min_length`` consecutive minutes of zero activity counts.  A limited
number of isolated non-zero spikes (epochs whose count ≤ ``spike_max_counts``)
are tolerated within the window without breaking the run.

Choi (2011)
-----------
Choi L, Liu Z, Matthews CE, Buchowski MS.
"Validation of accelerometer wear and nonwear time classification algorithm."
Med Sci Sports Exerc. 2011;43(2):357-364.

An improvement over Troiano that reduces false positives (e.g. misclassifying
sleep as non-wear).  The same spike tolerance as Troiano is applied, but a
candidate spike is only allowed to break the non-wear run if the
``window_size`` windows *immediately before and after* the spike are also
zero (or below ``spike_max_counts``).  This extra neighbourhood check means
genuine low-activity periods during sleep are much less likely to be flagged.

Both functions
--------------
* Accept a ``pd.Series`` of activity counts with a ``DatetimeIndex`` and infer
  the epoch length automatically.
* Return a binary ``pd.Series`` with the same index: **0 = non-wear**, **1 = wear**.
* Are compatible with the ``Mask`` API in ``circstudio.io.mask``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _epoch_minutes(activity: pd.Series) -> float:
    """Return the epoch length in minutes, inferred from the DatetimeIndex."""
    if activity.index.freq is not None:
        delta = pd.tseries.frequencies.to_offset(activity.index.freq)
        return pd.Timedelta(delta).total_seconds() / 60.0
    if len(activity) >= 2:
        delta = (activity.index[1] - activity.index[0]).total_seconds() / 60.0
        return delta
    return 1.0  # fallback: assume 1-minute epochs


def _min_length_epochs(min_length: str | int, epoch_minutes: float) -> int:
    """Convert *min_length* (offset string or plain int of epochs) to epochs."""
    if isinstance(min_length, str):
        return max(1, int(pd.Timedelta(min_length).total_seconds() / 60.0 / epoch_minutes))
    return int(min_length)


def _window_epochs(window_size: str | int, epoch_minutes: float) -> int:
    """Convert *window_size* to number of epochs."""
    if isinstance(window_size, str):
        return max(1, int(pd.Timedelta(window_size).total_seconds() / 60.0 / epoch_minutes))
    return int(window_size)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_nonwear_troiano(
    activity: pd.Series,
    min_length: "str | int" = "60min",
    spike_tolerance: int = 2,
    spike_max_counts: int = 100,
) -> pd.Series:
    """Troiano (2008) non-wear detection for actigraphy count data.

    A contiguous run of epochs is classified as *non-wear* when:

    1. The run contains at least ``min_length`` worth of epochs.
    2. Within the run, at most ``spike_tolerance`` epochs may have counts
       above zero, provided those counts do not exceed ``spike_max_counts``.
       Any such epoch is still considered part of the non-wear period.

    Parameters
    ----------
    activity : pd.Series
        Activity counts indexed by a ``DatetimeIndex``.
    min_length : str or int
        Minimum duration of a non-wear period.  Accepts a pandas offset
        string (e.g. ``"60min"``, ``"2h"``) or a plain integer number of
        epochs.  Default is ``"60min"``.
    spike_tolerance : int
        Maximum number of non-zero epochs allowed within a non-wear window.
        Default is 2 (NHANES default).
    spike_max_counts : int
        Counts at or below this value are treated as spikes rather than
        genuine activity.  Default is 100 (NHANES default).

    Returns
    -------
    pd.Series
        Binary series aligned to ``activity.index``: 0 = non-wear, 1 = wear.

    References
    ----------
    Troiano RP et al. Med Sci Sports Exerc. 2008;40(1):181-188.
    """
    epoch_min = _epoch_minutes(activity)
    min_epochs = _min_length_epochs(min_length, epoch_min)

    counts = activity.values.copy().astype(float)
    n = len(counts)
    mask = np.ones(n, dtype=int)

    i = 0
    while i < n:
        # Only start a candidate window when the current epoch is zero or a spike.
        if counts[i] > spike_max_counts:
            i += 1
            continue

        # Extend window as far as possible while the Troiano rule holds:
        # count zero epochs and spikes (non-zero but ≤ spike_max_counts).
        j = i
        n_spikes = 0
        while j < n:
            if counts[j] == 0 or counts[j] <= spike_max_counts:
                if 0 < counts[j] <= spike_max_counts:
                    n_spikes += 1
                if n_spikes > spike_tolerance:
                    break
                j += 1
            else:
                # Genuine activity: stop extending.
                break

        window_len = j - i
        if window_len >= min_epochs:
            mask[i:j] = 0

        # Advance: if we flagged a window, jump to j; otherwise step forward.
        i = j if window_len >= min_epochs else i + 1

    return pd.Series(mask, index=activity.index, name="nonwear_mask")


def detect_nonwear_choi(
    activity: pd.Series,
    min_length: "str | int" = "90min",
    window_size: "str | int" = "30min",
    spike_tolerance: int = 2,
    spike_max_counts: int = 100,
) -> pd.Series:
    """Choi (2011) non-wear detection for actigraphy count data.

    Extends the Troiano algorithm with a *neighbourhood check*: an isolated
    non-zero spike within a candidate non-wear window is only tolerated if
    the ``window_size`` epochs immediately *before* **and** after it are also
    zero (or below ``spike_max_counts``).  This prevents genuine low-activity
    episodes (e.g. restless sleep) from being absorbed into a non-wear period.

    Parameters
    ----------
    activity : pd.Series
        Activity counts indexed by a ``DatetimeIndex``.
    min_length : str or int
        Minimum duration of a non-wear period.  Default is ``"90min"``
        (Choi et al. recommendation).
    window_size : str or int
        Size of the upstream/downstream neighbourhood window used to validate
        spikes.  Default is ``"30min"``.
    spike_tolerance : int
        Maximum number of spikes allowed within a non-wear window.  Default 2.
    spike_max_counts : int
        Counts at or below this value are treated as spikes.  Default 100.

    Returns
    -------
    pd.Series
        Binary series: 0 = non-wear, 1 = wear.

    References
    ----------
    Choi L et al. Med Sci Sports Exerc. 2011;43(2):357-364.
    """
    epoch_min = _epoch_minutes(activity)
    min_epochs = _min_length_epochs(min_length, epoch_min)
    win_epochs = _window_epochs(window_size, epoch_min)

    counts = activity.values.copy().astype(float)
    n = len(counts)
    mask = np.ones(n, dtype=int)

    def _neighbourhood_ok(spike_idx: int) -> bool:
        """Return True if the windows before and after *spike_idx* are zero."""
        pre_start = max(0, spike_idx - win_epochs)
        post_end = min(n, spike_idx + win_epochs + 1)
        pre_window = counts[pre_start:spike_idx]
        post_window = counts[spike_idx + 1:post_end]
        pre_ok = all(c <= spike_max_counts for c in pre_window)
        post_ok = all(c <= spike_max_counts for c in post_window)
        return pre_ok and post_ok

    i = 0
    while i < n:
        if counts[i] > spike_max_counts:
            i += 1
            continue

        j = i
        n_spikes = 0
        valid = True
        while j < n:
            c = counts[j]
            if c == 0:
                j += 1
            elif c <= spike_max_counts:
                # Spike candidate — apply neighbourhood check.
                if not _neighbourhood_ok(j):
                    valid = False
                    break
                n_spikes += 1
                if n_spikes > spike_tolerance:
                    valid = False
                    break
                j += 1
            else:
                # Genuine activity.
                break

        window_len = j - i
        if valid and window_len >= min_epochs:
            mask[i:j] = 0
            i = j
        else:
            i += 1

    return pd.Series(mask, index=activity.index, name="nonwear_mask")
