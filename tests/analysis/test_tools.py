"""Internal helpers in ``circstudio.analysis.tools``."""

import numpy as np
import pandas as pd
import pytest

from circstudio.analysis.tools import (
    _activity_offset_time,
    _activity_onset_time,
    _average_daily_activity,
    _average_daily_total_activity,
    _binarize,
    _count_consecutive_values,
    _count_consecutive_zeros,
    _create_inactivity_mask,
    _impute_nan,
    _interval_maker,
    _light_exposure,
    _lmx,
    _offset_detection,
    _onset_detection,
    _resample,
    _shift_time_axis,
    _td_format,
    _transition_prob,
)
from helpers import assertions as A
from helpers import signals as S

EPOCHS_PER_DAY = 1440  # at the default 60 s sampling period


class TestAverageDailyActivity:
    def test_identical_days_return_that_day(self):
        """N identical days must average to exactly one of them."""
        series = S.squarewave(n_days=5)
        profile = _average_daily_activity(series, cyclic=False)
        assert len(profile) == EPOCHS_PER_DAY
        np.testing.assert_allclose(profile.values, series.values[:EPOCHS_PER_DAY])

    def test_cyclic_doubles_and_repeats(self):
        series = S.squarewave(n_days=3)
        profile = _average_daily_activity(series, cyclic=True)
        assert len(profile) == 2 * EPOCHS_PER_DAY
        first, second = profile.values[:EPOCHS_PER_DAY], profile.values[EPOCHS_PER_DAY:]
        np.testing.assert_allclose(first, second)

    def test_index_is_a_timedelta_range_starting_at_zero(self):
        profile = _average_daily_activity(S.squarewave(n_days=2), cyclic=False)
        assert isinstance(profile.index, pd.TimedeltaIndex)
        assert profile.index[0] == pd.Timedelta(0)
        assert profile.index[-1] == pd.Timedelta(23, unit="h") + pd.Timedelta(59, unit="m")

    def test_averaging_reduces_noise_but_preserves_mean(self):
        series = S.sinewave(n_days=14, noise_sd=20.0, mesor=100.0, amplitude=50.0)
        profile = _average_daily_activity(series, cyclic=False)
        assert profile.mean() == pytest.approx(series.mean(), rel=1e-9)
        assert profile.std() < series.std()


class TestShiftTimeAxis:
    def test_shift_produces_a_symmetric_axis(self):
        profile = _average_daily_activity(S.squarewave(n_days=3), cyclic=False)
        shifted = _shift_time_axis(profile, EPOCHS_PER_DAY // 2)
        assert shifted.index[0] == pd.Timedelta(-12, unit="h")
        assert len(shifted) == len(profile)

    def test_shifting_by_a_full_period_is_the_identity_on_values(self):
        profile = _average_daily_activity(S.squarewave(n_days=3), cyclic=False)
        shifted = _shift_time_axis(profile, EPOCHS_PER_DAY)
        np.testing.assert_allclose(np.sort(shifted.values), np.sort(profile.values))


class TestOnsetOffsetDetection:
    """``_onset_detection`` and ``_offset_detection`` are pure ratio functions."""

    def test_onset_detection_on_a_known_step(self):
        # mean(x[2:]) / mean(x[:2]) - 1 = 3/1 - 1 = 2
        x = np.array([1.0, 1.0, 3.0, 3.0])
        assert _onset_detection(x, whs=2) == pytest.approx(2.0)

    def test_offset_detection_is_the_reciprocal_arrangement(self):
        # mean(x[:2]) / mean(x[2:]) - 1 = 3/1 - 1 = 2
        x = np.array([3.0, 3.0, 1.0, 1.0])
        assert _offset_detection(x, whs=2) == pytest.approx(2.0)

    def test_flat_input_gives_zero_for_both(self):
        x = np.full(8, 5.0)
        assert _onset_detection(x, whs=4) == pytest.approx(0.0)
        assert _offset_detection(x, whs=4) == pytest.approx(0.0)

    def test_onset_detection_is_scale_invariant(self):
        """Multiplying the signal by k leaves the ratio unchanged."""
        x = np.array([1.0, 2.0, 6.0, 8.0])
        assert _onset_detection(x, whs=2) == pytest.approx(_onset_detection(10 * x, whs=2))


class TestActivityOnsetOffsetTime:
    def test_onset_lands_on_the_known_transition(self):
        """Square wave rises at 00:00 and falls at 12:00."""
        series = S.squarewave(n_days=7, on_hours=12.0)
        profile = _average_daily_activity(series, cyclic=True)
        onset = _activity_onset_time(profile, whs=60)
        offset = _activity_offset_time(profile, whs=60)

        # Both are Timedeltas on the cyclic (0..48 h) axis.
        onset_h = onset.total_seconds() / 3600 % 24
        offset_h = offset.total_seconds() / 3600 % 24
        assert onset_h == pytest.approx(0.0, abs=1.0) or onset_h == pytest.approx(24.0, abs=1.0)
        assert offset_h == pytest.approx(12.0, abs=1.0)

    def test_onset_and_offset_are_half_a_day_apart_for_a_symmetric_wave(self):
        series = S.squarewave(n_days=7, on_hours=12.0)
        profile = _average_daily_activity(series, cyclic=True)
        onset = _activity_onset_time(profile, whs=60)
        offset = _activity_offset_time(profile, whs=60)
        gap_hours = abs((offset - onset).total_seconds()) / 3600 % 24
        assert gap_hours == pytest.approx(12.0, abs=1.0)


class TestBinarize:
    def test_binarize_is_a_strict_comparison(self):
        series = S.as_series([0.0, 1.0, 2.0, 3.0])
        result = _binarize(series, threshold=1.0)
        # Strictly greater than the threshold.
        np.testing.assert_array_equal(result.values, [0, 0, 1, 1])

    def test_threshold_zero_keeps_only_positive_values(self):
        series = S.as_series([0.0, 0.5, 0.0, 10.0])
        result = _binarize(series, threshold=0)
        np.testing.assert_array_equal(result.values, [0, 1, 0, 1])

    def test_nan_is_preserved_not_binarized(self):
        series = S.as_series([1.0, np.nan, 5.0])
        result = _binarize(series, threshold=2.0)
        assert result.isna().sum() == 1
        assert np.isnan(result.iloc[1])

    def test_index_is_preserved(self):
        series = S.squarewave(n_days=1)
        result = _binarize(series, threshold=50)
        pd.testing.assert_index_equal(result.index, series.index)


class TestResample:
    def test_downsampling_conserves_the_total(self):
        series = S.squarewave(n_days=2)
        resampled = _resample(
            series, new_freq="10min", current_freq=series.index.freq, mask_inactivity=False
        )
        assert resampled.sum() == pytest.approx(series.sum(), rel=1e-9)

    def test_downsampling_reduces_length_by_the_expected_factor(self):
        series = S.squarewave(n_days=2)
        resampled = _resample(
            series, new_freq="10min", current_freq=series.index.freq, mask_inactivity=False
        )
        assert len(resampled) == pytest.approx(len(series) / 10, abs=1)

    def test_requesting_a_finer_frequency_returns_the_original(self):
        """Upsampling is refused: the function returns the input untouched."""
        series = S.squarewave(n_days=1)
        resampled = _resample(
            series, new_freq="30s", current_freq=series.index.freq, mask_inactivity=False
        )
        pd.testing.assert_series_equal(resampled, series)

    def test_no_current_freq_returns_the_original(self):
        series = S.squarewave(n_days=1)
        resampled = _resample(series, new_freq="1h", current_freq=None, mask_inactivity=False)
        pd.testing.assert_series_equal(resampled, series)


class TestAverageDailyTotalActivity:
    def test_uniform_recording_gives_counts_times_epochs_per_day(self):
        """Migrated from the legacy tests/test_adat.py."""
        series = S.flat(n_days=7, value=10.0)
        result = _average_daily_total_activity(series, rescale=False, exclude_ends=False)
        assert result == pytest.approx(10.0 * EPOCHS_PER_DAY)

    def test_exclude_ends_drops_first_and_last_day(self):
        series = S.flat(n_days=7, value=10.0)
        # Corrupt the first and last day only.
        series.iloc[:EPOCHS_PER_DAY] = 0.0
        series.iloc[-EPOCHS_PER_DAY:] = 0.0
        included = _average_daily_total_activity(series, rescale=False, exclude_ends=True)
        excluded = _average_daily_total_activity(series, rescale=False, exclude_ends=False)
        assert included == pytest.approx(10.0 * EPOCHS_PER_DAY)
        assert excluded < included

    def test_rescale_compensates_for_missing_epochs(self):
        """With NaN-masked epochs, rescaling should recover the uncorrupted daily total."""
        series = S.flat(n_days=7, value=10.0)
        # Drop 2 h from day 3; the count-based weight compensates
        series.loc["2020-01-04 08:00:00":"2020-01-04 09:59:00"] = np.nan
        rescaled = _average_daily_total_activity(series, rescale=True, exclude_ends=False)
        assert rescaled == pytest.approx(10.0 * EPOCHS_PER_DAY, rel=1e-9)


class TestLmx:
    def test_l5_falls_in_the_off_block_of_a_square_wave(self):
        series = S.squarewave(n_days=7, on_hours=12.0, low=0.0, high=100.0)
        t_start, value = _lmx(series, period="5h", lowest=True)
        assert value == pytest.approx(0.0, abs=1e-9)
        start_hour = t_start.total_seconds() / 3600 % 24
        # The 5 h window must fit entirely inside the 12:00-24:00 off block
        assert 12.0 <= start_hour <= 19.0

    def test_m10_falls_in_the_on_block_of_a_square_wave(self):
        series = S.squarewave(n_days=7, on_hours=12.0, low=0.0, high=100.0)
        t_start, value = _lmx(series, period="10h", lowest=False)
        assert value == pytest.approx(100.0, rel=1e-9)
        start_hour = t_start.total_seconds() / 3600 % 24
        assert 0.0 <= start_hour <= 2.0

    def test_l5_never_exceeds_m10(self):
        series = S.realistic_rest_activity(n_days=7)
        _, l5_value = _lmx(series, period="5h", lowest=True)
        _, m10_value = _lmx(series, period="10h", lowest=False)
        assert l5_value <= m10_value

    def test_flat_signal_gives_equal_l5_and_m10(self):
        series = S.flat(n_days=7, value=42.0)
        _, l5_value = _lmx(series, period="5h", lowest=True)
        _, m10_value = _lmx(series, period="10h", lowest=False)
        assert l5_value == pytest.approx(42.0)
        assert m10_value == pytest.approx(42.0)


class TestIntervalMaker:
    def test_seven_day_recording_split_into_daily_intervals(self):
        series = S.squarewave(n_days=7)
        intervals = _interval_maker(series.index, period="1D", verbose=False)
        assert len(intervals) == 6, (
            "a 7-day recording sampled at 1 min spans 6 days 23:59 between first "
            "and last timestamp, so only 6 whole days fit"
        )
        for start, stop in intervals:
            assert stop - start == pd.Timedelta(1, unit="D")

    def test_intervals_are_contiguous_and_non_overlapping(self):
        series = S.squarewave(n_days=7)
        intervals = _interval_maker(series.index, period="1D", verbose=False)
        for (_, stop), (next_start, _) in zip(intervals, intervals[1:]):
            assert stop == next_start

    def test_period_longer_than_recording_yields_no_intervals(self):
        series = S.squarewave(n_days=2)
        intervals = _interval_maker(series.index, period="30D", verbose=False)
        assert intervals == []


class TestConsecutiveValueCounting:
    def test_count_consecutive_values_on_a_hand_written_series(self):
        series = pd.Series([0, 0, 1, 1, 1, 0, 5])
        result = _count_consecutive_values(series)
        # Four runs: 0x2, 1x3, 0x1, 5x1
        assert result["counts"].tolist() == [2, 3, 1, 1]
        # state == 1 when the run sums to something positive
        assert result["state"].tolist() == [0, 1, 0, 1]

    def test_count_consecutive_zeros_returns_only_zero_runs_with_positions(self):
        series = pd.Series([0, 0, 1, 1, 1, 0, 5])
        result = _count_consecutive_zeros(series)
        assert result["counts"].tolist() == [2, 1]
        assert result["start"].tolist() == [0, 5]
        assert result["end"].tolist() == [2, 6]

    def test_all_zeros_is_a_single_run(self):
        result = _count_consecutive_zeros(pd.Series([0, 0, 0, 0]))
        assert result["counts"].tolist() == [4]

    def test_no_zeros_returns_empty(self):
        result = _count_consecutive_zeros(pd.Series([1, 2, 3]))
        assert len(result) == 0

    def test_run_lengths_sum_to_series_length(self):
        series = pd.Series(np.random.default_rng(0).integers(0, 2, 500))
        result = _count_consecutive_values(series)
        assert result["counts"].sum() == len(series)


class TestTransitionProb:
    def test_probabilities_are_in_the_unit_interval(self):
        series = _binarize(S.realistic_rest_activity(n_days=7), threshold=0)
        prob, weights = _transition_prob(series, from_zero_to_one=True)
        A.assert_within_range(prob.values, 0.0, 1.0, name="pRA")
        assert (weights.values > 0).all()

    def test_rest_to_activity_and_activity_to_rest_are_different(self):
        series = _binarize(S.realistic_rest_activity(n_days=7), threshold=0)
        pra, _ = _transition_prob(series, from_zero_to_one=True)
        par, _ = _transition_prob(series, from_zero_to_one=False)
        assert not pra.equals(par)

    def test_weights_are_sqrt_of_run_counts(self):
        """Weights must be positive and monotonically related to sample size."""
        series = _binarize(S.realistic_rest_activity(n_days=7), threshold=0)
        _, weights = _transition_prob(series, from_zero_to_one=True)
        assert np.isfinite(weights.values).all()


class TestTdFormat:
    @pytest.mark.parametrize(
        "delta,expected",
        [
            (pd.Timedelta(0), "00:00:00"),
            (pd.Timedelta(30, unit="s"), "00:00:30"),
            (pd.Timedelta(90, unit="m"), "01:30:00"),
            (pd.Timedelta(23, unit="h") + pd.Timedelta(59, unit="m"), "23:59:00"),
        ],
    )
    def test_sub_day_durations_format_correctly(self, delta, expected):
        assert _td_format(delta) == expected

    @pytest.mark.xfail(reason="_td_format drops whole days from durations over 24 h", strict=True)
    def test_durations_over_a_day_keep_their_days(self):
        assert _td_format(pd.Timedelta(25, unit="h") + pd.Timedelta(30, unit="m")) == "25:30:00"

    def test_the_truncation_is_reproducible(self):
        """Pin the current (buggy) behaviour so the xfail above is unambiguous."""
        assert _td_format(pd.Timedelta(25, unit="h") + pd.Timedelta(30, unit="m")) == "01:30:00"
        assert _td_format(pd.Timedelta(48, unit="h")) == "00:00:00"


class TestImputeNan:
    def test_imputation_fills_exactly_the_missing_positions(self):
        series = S.signal_with_gap(n_days=7, gap_start_epoch=100, gap_length_epochs=30)
        missing = series.isna()
        imputed = _impute_nan(series, method="mean")
        assert imputed.isna().sum() == 0
        # Untouched positions must be bit-identical.
        pd.testing.assert_series_equal(imputed[~missing], series[~missing])

    def test_imputed_values_come_from_the_same_time_of_day(self):
        """The imputation groups by clock time, so a periodic signal is restored."""
        series = S.squarewave(n_days=7).astype(float)
        truth = series.copy()
        series.iloc[100:130] = np.nan
        imputed = _impute_nan(series, method="mean")
        np.testing.assert_allclose(imputed.values, truth.values)

    @pytest.mark.parametrize("method", ["mean", "median"])
    def test_supported_methods_all_fill(self, method):
        series = S.signal_with_gap(n_days=3, gap_start_epoch=50, gap_length_epochs=20)
        assert _impute_nan(series, method=method).isna().sum() == 0


class TestLightExposure:
    def test_threshold_masks_values_below_it(self):
        light = S.light_squarewave(n_days=2, low=0.0, high=1000.0)
        masked = _light_exposure(light, threshold=500.0)
        assert masked.notna().sum() == (light >= 500.0).sum()
        assert (masked.dropna() >= 500.0).all()

    def test_no_threshold_and_no_window_returns_the_input(self):
        light = S.light_squarewave(n_days=2)
        pd.testing.assert_series_equal(_light_exposure(light), light)

    def test_specifying_only_one_bound_raises(self):
        light = S.light_squarewave(n_days=2)
        with pytest.raises(ValueError, match="Both start and stop"):
            _light_exposure(light, start_time="09:00:00")
        with pytest.raises(ValueError, match="Both start and stop"):
            _light_exposure(light, stop_time="17:00:00")

    @pytest.mark.xfail(
        reason="_light_exposure passes include_end, which pandas 2 removed",
        strict=True,
    )
    def test_time_window_restricts_to_the_requested_hours(self):
        light = S.light_squarewave(n_days=2, light_on_hours=16.0)
        windowed = _light_exposure(light, start_time="09:00:00", stop_time="17:00:00")
        assert windowed.index.hour.min() >= 9
        assert windowed.index.hour.max() < 17


class TestCreateInactivityMask:
    def test_duration_none_returns_none(self):
        series = S.signal_with_nonwear(n_days=1)
        assert _create_inactivity_mask(series, duration=None, threshold=1) is None

    def test_duration_minus_one_returns_an_all_ones_mask(self):
        series = S.signal_with_nonwear(n_days=1)
        mask = _create_inactivity_mask(series, duration=-1, threshold=1)
        assert (mask == 1).all()
        assert len(mask) == len(series)

    def test_a_zero_run_at_least_duration_long_is_masked(self):
        """Mask convention: 0 marks inactivity, 1 marks valid data."""
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=200, nonwear_length_epochs=90
        )
        mask = _create_inactivity_mask(series, duration=60, threshold=1)
        masked_runs = A.count_runs(mask.values == 0)
        assert masked_runs == [(200, 90)]

    def test_a_run_shorter_than_duration_is_not_masked(self):
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=200, nonwear_length_epochs=30
        )
        mask = _create_inactivity_mask(series, duration=60, threshold=1)
        assert (mask == 1).all()

    def test_boundary_run_of_exactly_duration_is_masked(self):
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=200, nonwear_length_epochs=60
        )
        mask = _create_inactivity_mask(series, duration=60, threshold=1)
        assert A.count_runs(mask.values == 0) == [(200, 60)]

    def test_signal_with_no_zeros_is_entirely_unmasked(self):
        series = S.flat(n_days=1, value=100.0)
        mask = _create_inactivity_mask(series, duration=60, threshold=1)
        assert (mask == 1).all()

    @pytest.mark.xfail(reason="An all-zero recording is reported as fully valid", strict=True)
    def test_entirely_zero_signal_is_entirely_masked(self):
        series = S.flat(n_days=1, value=0.0)
        mask = _create_inactivity_mask(series, duration=60, threshold=1)
        assert (mask == 0).all()

    def test_entirely_zero_signal_current_behaviour_is_pinned(self):
        """Pin the current (buggy) behaviour so the xfail above is unambiguous."""
        series = S.flat(n_days=1, value=0.0)
        mask = _create_inactivity_mask(series, duration=60, threshold=1)
        assert (mask == 1).all()

    def test_run_at_the_very_start_is_detected(self):
        series = S.signal_with_nonwear(n_days=1, nonwear_start_epoch=0, nonwear_length_epochs=90)
        mask = _create_inactivity_mask(series, duration=60, threshold=1)
        assert A.count_runs(mask.values == 0) == [(0, 90)]

    def test_run_at_the_very_end_is_detected(self):
        n = EPOCHS_PER_DAY
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=n - 90, nonwear_length_epochs=90
        )
        mask = _create_inactivity_mask(series, duration=60, threshold=1)
        runs = A.count_runs(mask.values == 0)
        assert runs == [(n - 90, 90)]
