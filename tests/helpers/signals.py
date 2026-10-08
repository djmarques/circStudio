"""Seeded synthetic signal generators with documented analytical ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default epoch length in seconds (native resolution of the bundled AWD files)
DEFAULT_SAMPLING_PERIOD = 60

# Default recording start: a Wednesday, away from a week boundary
DEFAULT_START = "2020-01-01 00:00:00"

SECONDS_PER_DAY = 86400


def datetime_index(
    start: str | pd.Timestamp = DEFAULT_START,
    n: int = 1440,
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
) -> pd.DatetimeIndex:
    """Regular ``DatetimeIndex`` of ``n`` points spaced ``sampling_period`` seconds."""
    return pd.date_range(
        start=start,
        periods=n,
        # Explicit unit avoids a NumPy DeprecationWarning from the helpers
        freq=pd.Timedelta(sampling_period, unit="s"),
    )


def as_series(
    data,
    start: str | pd.Timestamp = DEFAULT_START,
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
    name: str | None = None,
) -> pd.Series:
    """Wrap a raw array in a ``Series`` with a regular ``DatetimeIndex``."""
    data = np.asarray(data)
    index = datetime_index(start=start, n=len(data), sampling_period=sampling_period)
    return pd.Series(data=data, index=index, name=name)


def _epochs_per_day(sampling_period: int) -> int:
    if SECONDS_PER_DAY % sampling_period:
        raise ValueError(
            f"sampling_period={sampling_period}s does not divide a 24 h day evenly; "
            "the analytical ground truth in this module assumes it does."
        )
    return SECONDS_PER_DAY // sampling_period


# Periodic signals


def sinewave(
    n_days: int = 7,
    period_seconds: float = SECONDS_PER_DAY,
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
    amplitude: float = 100.0,
    mesor: float = 0.0,
    acrophase_hours: float = 0.0,
    noise_sd: float = 0.0,
    start: str | pd.Timestamp = DEFAULT_START,
    seed: int = 0,
) -> pd.Series:
    r"""Pure cosine, optionally with additive Gaussian noise.

    The signal is

    .. math:: x(t) = M + A \cos\!\left(\frac{2\pi}{T}(t - \varphi)\right)

    with :math:`M` = ``mesor``, :math:`A` = ``amplitude``, :math:`T` =
    ``period_seconds`` and :math:`\varphi` = ``acrophase_hours``.

    Ground truth
    ------------
    * Cosinor must recover ``mesor``, ``amplitude`` and an acrophase of
      ``acrophase_hours`` (the time of the *maximum*).
    * ``spectral_centroid`` sits at ``SECONDS_PER_DAY / period_seconds``
      cycles per day (1.0 for the default 24 h period).
    * Mean over a whole number of periods is exactly ``mesor``.
    * ``temporal_centroid`` of the daily profile is at ``acrophase_hours``.

    Note the *cosine* convention: at ``t == acrophase_hours`` the signal is at
    its maximum.  circStudio's ``Cosinor`` uses the same convention.
    """
    n = n_days * _epochs_per_day(sampling_period)
    t = np.arange(n) * sampling_period  # seconds since start
    phase = 2 * np.pi * (t - acrophase_hours * 3600.0) / period_seconds
    signal = mesor + amplitude * np.cos(phase)
    if noise_sd:
        rng = np.random.default_rng(seed)
        signal = signal + rng.normal(scale=noise_sd, size=n)
    return as_series(signal, start=start, sampling_period=sampling_period, name="activity")


def squarewave(
    n_days: int = 7,
    on_hours: float = 12.0,
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
    low: float = 0.0,
    high: float = 100.0,
    start: str | pd.Timestamp = DEFAULT_START,
) -> pd.Series:
    r"""Perfect rectangular wave: ``high`` for ``on_hours``, then ``low``, per day.

    The day starts *on*.  With the default ``start`` of midnight the signal is
    high from 00:00 to 12:00 and low from 12:00 to 24:00.

    Ground truth (for ``low == 0``, ``on_hours == 12``)
    ---------------------------------------------------
    Let ``E`` be epochs per day, ``D`` = ``n_days``, ``N = D * E``, ``A`` = ``high``.

    * **IS = 1.0 exactly.**  Every day is identical, so the between-day
      variance vanishes and interdaily stability saturates.
    * **IV** = ``8 * D / (N - 1)``.  Derivation: the denominator
      :math:`\sum_i (x_i - \bar x)^2 = N A^2/4` since :math:`\bar x = A/2` and
      every sample deviates by :math:`A/2`; the numerator counts squared
      successive differences, which are zero except at the 2 transitions per
      day, each contributing :math:`A^2`, giving :math:`2 D A^2`.  Then
      :math:`IV = N \cdot 2DA^2 / ((N-1) \cdot N A^2/4) = 8D/(N-1)`.
      For the defaults (7 days, 60 s epochs): ``56 / 10079``.
    * **L5 = low** -- the 5 least-active hours lie wholly inside the 12 h off block.
    * **M10 = high** -- the 10 most-active hours lie wholly inside the 12 h on block.
    * **RA = (high - low) / (high + low)**, i.e. exactly 1.0 when ``low == 0``.
    * Mean = ``low + (high - low) * on_hours / 24``.
    """
    epd = _epochs_per_day(sampling_period)
    on_epochs = int(round(on_hours * 3600 / sampling_period))
    if not 0 < on_epochs < epd:
        raise ValueError("on_hours must lie strictly between 0 and 24")
    one_day = np.concatenate([np.full(on_epochs, high), np.full(epd - on_epochs, low)])
    signal = np.tile(one_day, n_days)
    return as_series(signal, start=start, sampling_period=sampling_period, name="activity")


def two_tone(
    n_days: int = 7,
    periods_seconds: tuple[float, float] = (SECONDS_PER_DAY, SECONDS_PER_DAY / 6),
    amplitudes: tuple[float, float] = (100.0, 30.0),
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
    start: str | pd.Timestamp = DEFAULT_START,
) -> pd.Series:
    """Sum of two sinusoids with well-separated periods (default: 24 h + 4 h).

    Ground truth
    ------------
    * SSA with a window covering both periods must place each tone in its own
      eigentriple *pair*; w-correlation is ~1 within a pair and ~0 between pairs.
    * The power spectrum has exactly two peaks, at the two periods.
    * Mean over a whole number of both periods is 0.
    """
    n = n_days * _epochs_per_day(sampling_period)
    t = np.arange(n) * sampling_period
    signal = sum(
        a * np.sin(2 * np.pi * t / p) for p, a in zip(periods_seconds, amplitudes)
    )
    return as_series(signal, start=start, sampling_period=sampling_period, name="activity")


# Stochastic signals


def gaussian_noise(
    n_days: int = 7,
    mu: float = 100.0,
    sigma: float = 10.0,
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
    start: str | pd.Timestamp = DEFAULT_START,
    seed: int = 0,
) -> pd.Series:
    """White Gaussian noise.

    Ground truth
    ------------
    * DFA generalised Hurst exponent **H ~ 0.5** (uncorrelated process).
    * IS ~ 0 (no reproducible daily pattern).
    * ``spectral_centroid`` is flat -- no dominant frequency.
    """
    n = n_days * _epochs_per_day(sampling_period)
    rng = np.random.default_rng(seed)
    return as_series(
        rng.normal(mu, sigma, n), start=start, sampling_period=sampling_period, name="activity"
    )


def brown_noise(
    n_days: int = 7,
    sigma: float = 1.0,
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
    start: str | pd.Timestamp = DEFAULT_START,
    seed: int = 0,
) -> pd.Series:
    """Brownian motion -- the cumulative sum of white noise.

    Ground truth
    ------------
    * DFA generalised Hurst exponent **H ~ 1.5** (integrated white noise).
      This, paired with :func:`gaussian_noise`, is the canonical DFA validation.
    """
    n = n_days * _epochs_per_day(sampling_period)
    rng = np.random.default_rng(seed)
    return as_series(
        np.cumsum(rng.normal(0.0, sigma, n)),
        start=start,
        sampling_period=sampling_period,
        name="activity",
    )


def pink_noise(
    n_days: int = 7,
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
    start: str | pd.Timestamp = DEFAULT_START,
    seed: int = 0,
) -> pd.Series:
    """1/f noise, generated by spectral shaping of white noise.

    Ground truth
    ------------
    * DFA generalised Hurst exponent **H ~ 1.0** (long-range correlated).
    """
    n = n_days * _epochs_per_day(sampling_period)
    rng = np.random.default_rng(seed)
    white = rng.normal(size=n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    scaling = np.ones_like(freqs)
    scaling[1:] = 1.0 / np.sqrt(freqs[1:])
    shaped = np.fft.irfft(spectrum * scaling, n=n)
    return as_series(shaped, start=start, sampling_period=sampling_period, name="activity")


# Rest-activity signals


def realistic_rest_activity(
    n_days: int = 7,
    sleep_start_hour: float = 23.0,
    sleep_duration_hours: float = 8.0,
    n_awakenings: int = 2,
    awakening_minutes: int = 10,
    wake_counts: float = 200.0,
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
    start: str | pd.Timestamp = DEFAULT_START,
    seed: int = 0,
) -> pd.Series:
    """Rest-activity signal with a *known* sleep window and awakening count.

    A rectangular circadian envelope (wake = active, sleep = zero) modulated by
    Poisson counts, with ``n_awakenings`` bursts of activity of
    ``awakening_minutes`` inserted inside each night.

    Ground truth
    ------------
    * The main sleep period each night runs from ``sleep_start_hour`` for
      ``sleep_duration_hours``; every scoring algorithm should place the main
      sleep bout within tolerance of that window.
    * Each night is broken into exactly ``n_awakenings + 1`` sleep bouts.
    * Total sleep time per night ~ ``sleep_duration_hours`` minus
      ``n_awakenings * awakening_minutes``.
    * Activity is non-negative everywhere (counts), as real actigraphy is.

    The awakenings are placed on a deterministic grid strictly inside the sleep
    window, never touching its edges, so onset/offset detection is unambiguous.
    """
    epd = _epochs_per_day(sampling_period)
    epochs_per_hour = 3600 // sampling_period
    rng = np.random.default_rng(seed)

    n = n_days * epd
    signal = rng.poisson(wake_counts, n).astype(float)

    sleep_start_epoch = int(round(sleep_start_hour * epochs_per_hour))
    sleep_len = int(round(sleep_duration_hours * epochs_per_hour))
    awake_len = int(round(awakening_minutes * 60 / sampling_period))

    for day in range(n_days):
        begin = day * epd + sleep_start_epoch
        # Zero the whole sleep window (it may run past midnight into the next day).
        for offset in range(sleep_len):
            idx = begin + offset
            if idx < n:
                signal[idx] = 0.0
        # Insert awakenings on an interior grid: at fractions k/(n+1) of the window.
        for k in range(1, n_awakenings + 1):
            centre = begin + int(round(sleep_len * k / (n_awakenings + 1)))
            for offset in range(awake_len):
                idx = centre + offset
                if idx < n:
                    signal[idx] = float(rng.poisson(wake_counts))

    return as_series(signal, start=start, sampling_period=sampling_period, name="activity")


def flat(
    n_days: int = 7,
    value: float = 10.0,
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
    start: str | pd.Timestamp = DEFAULT_START,
) -> pd.Series:
    """Constant signal -- the degenerate case.

    Ground truth
    ------------
    * IV = 0 (no successive variability).
    * RA is undefined when ``value == 0``; otherwise L5 == M10 == value and RA == 0.
    * DFA is degenerate (zero fluctuation at every scale): routines must return
      a documented degenerate value or raise, not produce a silent NaN.
    """
    n = n_days * _epochs_per_day(sampling_period)
    return as_series(
        np.full(n, float(value)), start=start, sampling_period=sampling_period, name="activity"
    )


def single_spike(
    n_days: int = 1,
    spike_index: int | None = None,
    spike_value: float = 1000.0,
    baseline: float = 0.0,
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
    start: str | pd.Timestamp = DEFAULT_START,
) -> pd.Series:
    """Flat baseline with one non-zero sample.

    Ground truth
    ------------
    * ``get_extremum('max')`` returns exactly the spike timestamp.
    * ``get_time_barycentre`` returns the spike timestamp (all mass is there)
      when ``baseline == 0``.
    * Exactly one run of non-zero values, of length 1.
    """
    n = n_days * _epochs_per_day(sampling_period)
    if spike_index is None:
        spike_index = n // 2
    signal = np.full(n, float(baseline))
    signal[spike_index] = spike_value
    return as_series(signal, start=start, sampling_period=sampling_period, name="activity")


# Signals with defects (gaps, non-wear)


def signal_with_gap(
    n_days: int = 7,
    gap_start_epoch: int = 1000,
    gap_length_epochs: int = 120,
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
    start: str | pd.Timestamp = DEFAULT_START,
    seed: int = 0,
) -> pd.Series:
    """Poisson activity with one contiguous block of NaN.

    Ground truth
    ------------
    * Exactly ``gap_length_epochs`` NaNs, starting at ``gap_start_epoch``.
    * Imputation must fill exactly those positions and leave the rest bit-identical.
    """
    epd = _epochs_per_day(sampling_period)
    n = n_days * epd
    rng = np.random.default_rng(seed)
    signal = rng.poisson(100, n).astype(float)
    signal[gap_start_epoch : gap_start_epoch + gap_length_epochs] = np.nan
    return as_series(signal, start=start, sampling_period=sampling_period, name="activity")


def signal_with_nonwear(
    n_days: int = 2,
    nonwear_start_epoch: int = 600,
    nonwear_length_epochs: int = 90,
    spike_offsets: tuple[int, ...] = (),
    spike_value: float = 50.0,
    baseline_counts: float = 500.0,
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
    start: str | pd.Timestamp = DEFAULT_START,
    seed: int = 0,
) -> pd.Series:
    """Active signal containing one zero-run of *exactly* known length.

    ``spike_offsets`` inserts non-zero samples *inside* the zero run, at the
    given offsets from its start, each of magnitude ``spike_value``.  This is
    how the Troiano/Choi spike-tolerance rules are exercised: the run is
    otherwise long enough to qualify as non-wear, and the question is whether a
    given spike pattern breaks it.

    Ground truth
    ------------
    * Outside the run every sample is strictly positive (never accidentally zero).
    * Outside the run every sample **exceeds 100 counts**, the default
      ``spike_max_counts`` of both non-wear algorithms.  This matters: a
      baseline that dipped to <= 100 would be silently reclassified as a
      *spike* rather than genuine activity, and the hand-counted ground truth
      below would be wrong.  ``baseline_counts=500`` puts the Poisson draw
      about 20 standard deviations clear of that threshold.
    * The run occupies ``[nonwear_start_epoch, nonwear_start_epoch + nonwear_length_epochs)``.
    """
    epd = _epochs_per_day(sampling_period)
    n = n_days * epd
    rng = np.random.default_rng(seed)
    # +1 guarantees strict positivity so the only zeros are the ones we insert.
    signal = rng.poisson(baseline_counts, n).astype(float) + 1.0
    signal[nonwear_start_epoch : nonwear_start_epoch + nonwear_length_epochs] = 0.0
    for offset in spike_offsets:
        if 0 <= offset < nonwear_length_epochs:
            signal[nonwear_start_epoch + offset] = spike_value
    return as_series(signal, start=start, sampling_period=sampling_period, name="activity")


# Light signals


def light_squarewave(
    n_days: int = 7,
    light_on_hours: float = 16.0,
    low: float = 0.0,
    high: float = 1000.0,
    sampling_period: int = DEFAULT_SAMPLING_PERIOD,
    start: str | pd.Timestamp = DEFAULT_START,
) -> pd.Series:
    """Rectangular light/dark cycle in lux, for entrainment tests.

    Ground truth
    ------------
    * Exactly ``light_on_hours`` of ``high`` per 24 h, the rest ``low``.
    * A circadian model driven by this must entrain to exactly 24 h.
    """
    series = squarewave(
        n_days=n_days,
        on_hours=light_on_hours,
        sampling_period=sampling_period,
        low=low,
        high=high,
        start=start,
    )
    return series.rename("light")
