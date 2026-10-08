"""Non-parametric rhythm metrics and their periodic variants."""

import numpy as np
import pandas as pd
import pytest

from circstudio.analysis import (
    IS,
    IV,
    ISp,
    IVp,
    TAT,
    TATp,
    VAT,
    adat,
    adatp,
    daily_profile,
    daily_profile_auc,
    get_extremum,
    get_time_barycentre,
    l5,
    l5p,
    lmx,
    m10,
    m10p,
    ra,
    rap,
    spectral_centroid,
    temporal_centroid,
)
from helpers import assertions as A
from helpers import signals as S

EPOCHS_PER_DAY = 1440


# Interdaily stability


class TestInterdailyStability:
    @pytest.mark.xfail(
        reason="IS uses ddof=1 for both variances, so a perfectly repeating series exceeds 1",
        strict=True,
    )
    def test_perfectly_repeating_pattern_gives_one(self):
        """Every day identical -> between-day variance equals total variance."""
        series = S.squarewave(n_days=7)
        assert IS(series) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("n_days", [2, 5, 7, 14])
    def test_bessel_correction_bias_is_exactly_predictable(self, n_days):
        """Pin the current behaviour and show it is the ddof mismatch, not noise."""
        series = S.squarewave(n_days=n_days)
        epochs_per_day = EPOCHS_PER_DAY
        n = n_days * epochs_per_day
        predicted = (n - 1) / (n_days * (epochs_per_day - 1))
        assert IS(series) == pytest.approx(predicted, rel=1e-12)
        assert IS(series) > 1.0

    def test_white_noise_gives_near_zero(self):
        """No reproducible daily pattern -> IS collapses toward 0."""
        series = S.gaussian_noise(n_days=14, seed=0)
        assert IS(series) == pytest.approx(0.0, abs=0.15)

    @pytest.mark.xfail(reason="Same ddof bug: the square wave gives IS = 1.0006", strict=True)
    def test_lies_in_the_unit_interval(self):
        for series in (
            S.squarewave(n_days=7),
            S.gaussian_noise(n_days=7),
            S.realistic_rest_activity(n_days=7),
            S.sinewave(n_days=7, mesor=100.0, amplitude=50.0),
        ):
            A.assert_within_range(IS(series), 0.0, 1.0, name="IS")

    def test_realistic_signals_stay_in_range_despite_the_bug(self):
        """The breach only shows on near-perfect rhythms; real data is safe."""
        for series in (
            S.gaussian_noise(n_days=7),
            S.realistic_rest_activity(n_days=7),
        ):
            A.assert_within_range(IS(series), 0.0, 1.0, name="IS")

    def test_adding_noise_reduces_stability(self):
        clean = S.sinewave(n_days=14, mesor=100.0, amplitude=50.0, noise_sd=0.0)
        noisy = S.sinewave(n_days=14, mesor=100.0, amplitude=50.0, noise_sd=40.0, seed=1)
        assert IS(noisy) < IS(clean)

    def test_invariant_under_scaling(self):
        """IS is a variance ratio, so multiplying the signal cancels out."""
        series = S.realistic_rest_activity(n_days=7)
        assert IS(series * 7.5) == pytest.approx(IS(series), rel=1e-9)

    def test_invariant_under_offset(self):
        series = S.realistic_rest_activity(n_days=7)
        assert IS(series + 1000.0) == pytest.approx(IS(series), rel=1e-9)

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_constant_signal_is_undefined(self):
        """0/0 for a constant signal: NaN, without a division warning."""
        assert np.isnan(IS(S.flat(n_days=7, value=10.0)))


# Intradaily variability


class TestIntradailyVariability:
    def test_closed_form_on_a_square_wave(self):
        r"""IV = sum(diff^2) / sum((x - xbar)^2) for this implementation."""
        n_days, high = 7, 100.0
        series = S.squarewave(n_days=n_days, low=0.0, high=high)
        n = len(series)
        expected = 4 * (2 * n_days - 1) / n
        assert IV(series) == pytest.approx(expected, rel=1e-9)

    def test_implementation_differs_from_the_textbook_formula_by_one_epoch(self):
        """Witting (1990) defines IV with a leading N/(N-1) factor."""
        series = S.squarewave(n_days=7)
        x = series.values
        n = len(x)
        textbook = n * np.sum(np.diff(x) ** 2) / ((n - 1) * np.sum((x - x.mean()) ** 2))
        assert IV(series) == pytest.approx(textbook * (n - 1) / n, rel=1e-12)
        assert IV(series) == pytest.approx(textbook, rel=2 / n)

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_constant_signal_is_undefined(self):
        """0/0 for a constant signal: NaN, without a division warning."""
        assert np.isnan(IV(S.flat(n_days=7, value=10.0)))

    def test_noise_is_more_fragmented_than_a_square_wave(self):
        assert IV(S.gaussian_noise(n_days=7)) > IV(S.squarewave(n_days=7))

    def test_is_non_negative(self):
        for series in (
            S.squarewave(n_days=7),
            S.gaussian_noise(n_days=7),
            S.realistic_rest_activity(n_days=7),
        ):
            assert IV(series) >= 0.0

    def test_invariant_under_scaling_and_offset(self):
        series = S.realistic_rest_activity(n_days=7)
        base = IV(series)
        assert IV(series * 3.0) == pytest.approx(base, rel=1e-9)
        assert IV(series + 500.0) == pytest.approx(base, rel=1e-9)


# L5 / M10 / RA


class TestL5M10RA:
    def test_square_wave_l5_is_the_off_level(self):
        series = S.squarewave(n_days=7, low=0.0, high=100.0)
        onset, value = l5(series)
        assert value == pytest.approx(0.0, abs=1e-9)
        assert isinstance(onset, pd.Timedelta)

    def test_square_wave_m10_is_the_on_level(self):
        series = S.squarewave(n_days=7, low=0.0, high=100.0)
        onset, value = m10(series)
        assert value == pytest.approx(100.0, rel=1e-9)

    def test_relative_amplitude_is_one_when_the_trough_is_zero(self):
        series = S.squarewave(n_days=7, low=0.0, high=100.0)
        assert ra(series) == pytest.approx(1.0, rel=1e-9)

    def test_relative_amplitude_closed_form_for_a_raised_trough(self):
        """RA = (M10 - L5) / (M10 + L5) = (100-20)/(100+20)."""
        series = S.squarewave(n_days=7, low=20.0, high=100.0)
        assert ra(series) == pytest.approx(80 / 120, rel=1e-9)

    def test_flat_signal_has_zero_relative_amplitude(self):
        series = S.flat(n_days=7, value=42.0)
        assert ra(series) == pytest.approx(0.0, abs=1e-9)

    def test_l5_never_exceeds_m10(self):
        for series in (
            S.realistic_rest_activity(n_days=7),
            S.sinewave(n_days=7, mesor=100.0, amplitude=50.0),
            S.gaussian_noise(n_days=7),
        ):
            assert l5(series)[1] <= m10(series)[1]

    def test_relative_amplitude_within_minus_one_and_one(self):
        for series in (
            S.realistic_rest_activity(n_days=7),
            S.sinewave(n_days=7, mesor=100.0, amplitude=50.0),
        ):
            A.assert_within_range(ra(series), -1.0, 1.0, name="RA")

    def test_l5_and_m10_scale_linearly(self):
        series = S.realistic_rest_activity(n_days=7)
        assert l5(series * 3.0)[1] == pytest.approx(3.0 * l5(series)[1], rel=1e-9)
        assert m10(series * 3.0)[1] == pytest.approx(3.0 * m10(series)[1], rel=1e-9)

    def test_relative_amplitude_is_scale_invariant(self):
        series = S.realistic_rest_activity(n_days=7)
        assert ra(series * 3.0) == pytest.approx(ra(series), rel=1e-9)

    def test_lmx_generalises_l5_and_m10(self):
        """``lmx`` with the right length must reproduce ``l5``/``m10`` exactly."""
        series = S.realistic_rest_activity(n_days=7)
        assert lmx(series, length="5h", lowest=True) == l5(series)
        assert lmx(series, length="10h", lowest=False) == m10(series)


# ADAT


class TestAdat:
    def test_uniform_recording(self):
        """Migrated from legacy tests/test_adat.py."""
        series = S.flat(n_days=7, value=10.0)
        assert adat(series, rescale=False, exclude_ends=False) == pytest.approx(
            10.0 * EPOCHS_PER_DAY
        )

    def test_scales_linearly_with_counts(self):
        series = S.flat(n_days=7, value=10.0)
        doubled = S.flat(n_days=7, value=20.0)
        assert adat(doubled, rescale=False) == pytest.approx(
            2 * adat(series, rescale=False)
        )

    def test_exclude_ends_ignores_partial_first_and_last_days(self):
        series = S.flat(n_days=7, value=10.0)
        series.iloc[:EPOCHS_PER_DAY] = 0.0
        series.iloc[-EPOCHS_PER_DAY:] = 0.0
        assert adat(series, rescale=False, exclude_ends=True) == pytest.approx(
            10.0 * EPOCHS_PER_DAY
        )


# The "per period" variants -- the strongest self-consistency check available


class TestPeriodicVariantsMatchWholeSeries:
    """Each ``Xp`` over a window must equal ``X`` computed on that window alone."""

    @pytest.fixture(scope="class")
    @classmethod
    def fortnight(cls):
        return S.realistic_rest_activity(n_days=14, seed=3)

    @staticmethod
    def _windows(series, period="7D"):
        from circstudio.analysis.tools import _interval_maker

        return _interval_maker(series.index, period, False)

    def test_isp_matches_is_per_window(self, fortnight):
        windows = self._windows(fortnight)
        expected = [IS(fortnight[a:b]) for a, b in windows]
        assert ISp(fortnight, period="7D") == pytest.approx(expected)

    def test_ivp_matches_iv_per_window(self, fortnight):
        windows = self._windows(fortnight)
        expected = [IV(fortnight[a:b]) for a, b in windows]
        assert IVp(fortnight, period="7D") == pytest.approx(expected)

    def test_l5p_matches_l5_per_window(self, fortnight):
        """``l5`` returns ``(onset, value)``; ``l5p`` returns values only."""
        windows = self._windows(fortnight)
        expected = [l5(fortnight[a:b])[1] for a, b in windows]
        assert l5p(fortnight, period="7D") == pytest.approx(expected)

    def test_m10p_matches_m10_per_window(self, fortnight):
        windows = self._windows(fortnight)
        expected = [m10(fortnight[a:b])[1] for a, b in windows]
        assert m10p(fortnight, period="7D") == pytest.approx(expected)

    def test_periodic_variants_drop_the_onset_times(self):
        """Recorded as an API inconsistency, not a bug."""
        series = S.realistic_rest_activity(n_days=14)
        assert all(np.isscalar(v) or isinstance(v, float) for v in l5p(series))
        assert isinstance(l5(series), tuple) and len(l5(series)) == 2

    def test_rap_matches_ra_per_window(self, fortnight):
        windows = self._windows(fortnight)
        expected = [ra(fortnight[a:b]) for a, b in windows]
        assert rap(fortnight, period="7D") == pytest.approx(expected)

    def test_adatp_matches_adat_per_window(self, fortnight):
        windows = self._windows(fortnight)
        expected = [adat(fortnight[a:b], rescale=True) for a, b in windows]
        assert adatp(fortnight, period="7D", rescale=True) == pytest.approx(expected)

    def test_tatp_is_per_calendar_day_not_per_period(self, fortnight):
        """``TATp`` breaks the naming convention of every other ``p`` variant."""
        result = TATp(fortnight, threshold=100)
        assert len(result) == 14, "one entry per calendar date"

        expected = [
            TAT(fortnight[fortnight.index.date == day], threshold=100)
            for day in sorted(set(fortnight.index.date))
        ]
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected))

    def test_tatp_daily_totals_sum_to_the_whole_recording_total(self, fortnight):
        assert TATp(fortnight, threshold=100).sum() == TAT(fortnight, threshold=100)


class TestPeriodicVariantWindowing:
    def test_a_fortnight_yields_one_whole_weekly_window(self):
        """Only one full 7-day window fits in 14 days: ``_interval_maker`` floors."""
        series = S.realistic_rest_activity(n_days=14)
        assert len(ISp(series, period="7D")) == 1

    def test_fifteen_days_yields_two_weekly_windows(self):
        series = S.realistic_rest_activity(n_days=15)
        assert len(ISp(series, period="7D")) == 2

    def test_daily_windows(self):
        series = S.realistic_rest_activity(n_days=7)
        # 7 days at 1 min: first-to-last span is 6 d 23:59, so 6 whole days fit.
        assert len(ISp(series, period="1D")) == 6

    def test_period_longer_than_recording_yields_no_windows(self):
        series = S.realistic_rest_activity(n_days=3)
        assert ISp(series, period="30D") == []

    def test_verbose_does_not_change_the_numbers(self, capsys):
        series = S.realistic_rest_activity(n_days=14)
        quiet = ISp(series, period="7D", verbose=False)
        loud = ISp(series, period="7D", verbose=True)
        assert quiet == pytest.approx(loud)
        assert capsys.readouterr().out != "", "verbose=True should print something"


# Time above threshold


class TestTimeAboveThreshold:
    def test_counts_epochs_at_or_above_threshold(self):
        """12 h on per day at 1 min epochs = 720 epochs/day."""
        series = S.squarewave(n_days=3, on_hours=12.0, low=0.0, high=100.0)
        assert TAT(series, threshold=50) == 3 * 720

    def test_threshold_above_the_maximum_counts_nothing(self):
        series = S.squarewave(n_days=3, low=0.0, high=100.0)
        assert TAT(series, threshold=1000) == 0

    def test_no_threshold_counts_every_epoch(self):
        series = S.squarewave(n_days=3)
        assert TAT(series) == len(series)

    def test_minute_output_format(self):
        series = S.squarewave(n_days=3, on_hours=12.0, low=0.0, high=100.0)
        assert TAT(series, threshold=50, oformat="minute") == pytest.approx(3 * 720)

    def test_timedelta_output_format(self):
        series = S.squarewave(n_days=3, on_hours=12.0, low=0.0, high=100.0)
        result = TAT(series, threshold=50, oformat="timedelta")
        assert result == pd.Timedelta(3 * 12, unit="h")

    def test_unsupported_output_format_raises(self):
        series = S.squarewave(n_days=1)
        with pytest.raises(ValueError, match="not supported"):
            TAT(series, oformat="hours")

    def test_monotonic_in_threshold(self):
        series = S.realistic_rest_activity(n_days=3)
        counts = [TAT(series, threshold=t) for t in (0, 50, 100, 200, 400)]
        A.assert_monotonic(counts, increasing=False)

    def test_vat_masks_rather_than_counts(self):
        """VAT returns the values above threshold, not their number."""
        series = S.squarewave(n_days=1, low=0.0, high=100.0)
        result = VAT(series, threshold=50)
        assert isinstance(result, pd.Series)
        assert result.notna().sum() == TAT(series, threshold=50)
        assert (result.dropna() >= 50).all()


# Profiles and centroids


class TestDailyProfile:
    def test_identical_days_reproduce_that_day(self):
        series = S.squarewave(n_days=5)
        profile = daily_profile(series, cyclic=False)
        assert len(profile) == EPOCHS_PER_DAY
        np.testing.assert_allclose(profile.values, series.values[:EPOCHS_PER_DAY])

    def test_cyclic_profile_is_twice_as_long_and_repeats(self):
        series = S.squarewave(n_days=3)
        profile = daily_profile(series, cyclic=True)
        assert len(profile) == 2 * EPOCHS_PER_DAY
        np.testing.assert_allclose(
            profile.values[:EPOCHS_PER_DAY], profile.values[EPOCHS_PER_DAY:]
        )

    def test_whs_is_ignored_unless_time_origin_is_an_onset_keyword(self):
        """``whs`` is a detection window half-size, not a smoother."""
        series = S.realistic_rest_activity(n_days=7)
        default = daily_profile(series, whs="1h")
        wider = daily_profile(series, whs="4h")
        pd.testing.assert_series_equal(default, wider)

    def test_time_origin_shifts_the_profile_onto_a_symmetric_axis(self):
        series = S.squarewave(n_days=7, on_hours=12.0)
        shifted = daily_profile(series, time_origin="12:00:00")
        assert shifted.index[0] == pd.Timedelta(-12, unit="h")
        assert len(shifted) == EPOCHS_PER_DAY

    def test_time_origin_preserves_the_multiset_of_values(self):
        series = S.squarewave(n_days=7, on_hours=12.0)
        plain = daily_profile(series)
        shifted = daily_profile(series, time_origin="06:00:00")
        np.testing.assert_allclose(np.sort(shifted.values), np.sort(plain.values))

    def test_unsupported_time_origin_raises(self):
        series = S.realistic_rest_activity(n_days=3)
        with pytest.raises(ValueError, match="not supported"):
            daily_profile(series, time_origin="halfway")

    def test_auc_over_the_whole_day_is_the_profile_total(self):
        series = S.squarewave(n_days=5, low=0.0, high=100.0)
        auc = daily_profile_auc(series)
        assert auc == pytest.approx(daily_profile(series).sum(), rel=1e-9)


class TestCentroidsAndExtrema:
    def test_get_extremum_finds_the_spike(self):
        series = S.single_spike(n_days=1, spike_index=600, spike_value=1000.0)
        timestamp, value = get_extremum(series, "max")
        assert value == 1000.0
        assert timestamp == series.index[600]

    def test_get_extremum_min(self):
        series = S.single_spike(n_days=1, spike_index=600, baseline=5.0)
        _, value = get_extremum(series, "min")
        assert value == 5.0

    def test_get_extremum_rejects_bad_argument(self):
        with pytest.raises(ValueError, match='"min" or "max"'):
            get_extremum(S.flat(n_days=1), "middle")

    def test_time_barycentre_of_a_single_spike_is_that_epoch(self):
        series = S.single_spike(n_days=1, spike_index=600, baseline=0.0)
        assert get_time_barycentre(series) == pytest.approx(600.0)

    def test_time_barycentre_of_a_flat_day_is_the_midpoint(self):
        series = S.flat(n_days=1, value=1.0)
        assert get_time_barycentre(series) == pytest.approx((EPOCHS_PER_DAY - 1) / 2)

    def test_temporal_centroid_of_a_symmetric_signal_is_its_midpoint(self):
        series = S.flat(n_days=2, value=1.0)
        centroid = temporal_centroid(series)
        midpoint = series.index[0] + (series.index[-1] - series.index[0]) / 2
        assert abs(centroid - midpoint) < pd.Timedelta(1, unit="m")

    def test_spectral_centroid_of_a_daily_sinusoid_sits_near_one_cycle_per_day(self):
        series = S.sinewave(n_days=8, period_seconds=86400, amplitude=100.0, mesor=0.0)
        centroid_hz = spectral_centroid(series)
        assert centroid_hz == pytest.approx(1 / 86400, rel=0.25)

    def test_spectral_centroid_rises_with_faster_oscillation(self):
        slow = S.sinewave(n_days=8, period_seconds=86400)
        fast = S.sinewave(n_days=8, period_seconds=86400 / 6)
        assert spectral_centroid(fast) > spectral_centroid(slow)

    def test_spectral_centroid_of_an_all_zero_signal_returns_none(self):
        series = S.flat(n_days=1, value=0.0)
        assert spectral_centroid(series) is None
