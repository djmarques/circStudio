"""Non-wear detection (Troiano and Choi) in ``circstudio.preprocessing.nonwear``."""

import numpy as np
import pandas as pd
import pytest

from circstudio.preprocessing import detect_nonwear_choi, detect_nonwear_troiano
from circstudio.preprocessing.nonwear import (
    _epoch_minutes,
    _min_length_epochs,
    _window_epochs,
)
from helpers import assertions as A
from helpers import signals as S


def masked_runs(mask: pd.Series):
    """``(start, length)`` of every non-wear run in a mask."""
    return A.count_runs(mask.values == 0)


class TestEpochConversionHelpers:
    @pytest.mark.parametrize(
        "sampling_period,expected_minutes",
        [(15, 0.25), (30, 0.5), (60, 1.0), (300, 5.0)],
    )
    def test_epoch_minutes_from_index_freq(self, sampling_period, expected_minutes):
        series = S.squarewave(n_days=1, sampling_period=sampling_period)
        assert _epoch_minutes(series) == pytest.approx(expected_minutes)

    def test_epoch_minutes_falls_back_to_index_spacing(self):
        """When freq is unset the helper measures the first gap instead."""
        series = S.squarewave(n_days=1, sampling_period=60)
        series.index.freq = None
        assert _epoch_minutes(series) == pytest.approx(1.0)

    def test_epoch_minutes_single_sample_falls_back_to_one_minute(self):
        series = pd.Series([1.0], index=pd.to_datetime(["2020-01-01"]))
        assert _epoch_minutes(series) == 1.0

    @pytest.mark.parametrize(
        "min_length,epoch_minutes,expected",
        [
            ("60min", 1.0, 60),
            ("60min", 0.5, 120),  # 30 s epochs -> twice as many
            ("60min", 5.0, 12),  # 5 min epochs
            ("90min", 1.0, 90),
            ("2h", 1.0, 120),
            ("1h", 0.25, 240),  # 15 s epochs
        ],
    )
    def test_min_length_offset_strings_convert_to_epochs(
        self, min_length, epoch_minutes, expected
    ):
        assert _min_length_epochs(min_length, epoch_minutes) == expected

    def test_min_length_integer_is_taken_as_epochs_verbatim(self):
        """A bare int is a number of *epochs*, not minutes -- no conversion."""
        assert _min_length_epochs(45, 5.0) == 45
        assert _min_length_epochs(45, 0.25) == 45

    def test_min_length_shorter_than_one_epoch_clamps_to_one(self):
        """Rounding rule: truncation toward zero, floored at 1 epoch."""
        assert _min_length_epochs("1min", 5.0) == 1

    def test_non_integer_conversion_truncates(self):
        """70 min at 5 min epochs = 14 exactly; 72 min truncates to 14, not 15."""
        assert _min_length_epochs("70min", 5.0) == 14
        assert _min_length_epochs("72min", 5.0) == 14

    @pytest.mark.parametrize(
        "window_size,epoch_minutes,expected",
        [("30min", 1.0, 30), ("30min", 0.5, 60), ("30min", 5.0, 6), (10, 1.0, 10)],
    )
    def test_window_epochs_conversion(self, window_size, epoch_minutes, expected):
        assert _window_epochs(window_size, epoch_minutes) == expected


class TestTroianoLengthRule:
    """The core rule: >= min_length consecutive zero epochs."""

    def test_run_of_exactly_min_length_is_flagged(self):
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=300, nonwear_length_epochs=60
        )
        mask = detect_nonwear_troiano(series, min_length="60min", spike_tolerance=0)
        assert masked_runs(mask) == [(300, 60)]

    def test_run_one_epoch_short_is_not_flagged(self):
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=300, nonwear_length_epochs=59
        )
        mask = detect_nonwear_troiano(series, min_length="60min", spike_tolerance=0)
        assert masked_runs(mask) == []

    def test_longer_run_is_flagged_in_full(self):
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=300, nonwear_length_epochs=200
        )
        mask = detect_nonwear_troiano(series, min_length="60min", spike_tolerance=0)
        assert masked_runs(mask) == [(300, 200)]

    def test_two_separate_runs_are_both_flagged(self):
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=200, nonwear_length_epochs=90
        )
        series.iloc[600:700] = 0.0
        mask = detect_nonwear_troiano(series, min_length="60min", spike_tolerance=0)
        assert masked_runs(mask) == [(200, 90), (600, 100)]

    def test_mask_is_binary_and_aligned(self):
        series = S.signal_with_nonwear(n_days=2)
        mask = detect_nonwear_troiano(series)
        A.assert_is_binary(mask)
        pd.testing.assert_index_equal(mask.index, series.index)
        assert mask.name == "nonwear_mask"


class TestTroianoSpikeRule:
    """Spikes: non-zero epochs of <= spike_max_counts inside a candidate run."""

    def test_spikes_within_tolerance_do_not_break_the_run(self):
        series = S.signal_with_nonwear(
            n_days=1,
            nonwear_start_epoch=300,
            nonwear_length_epochs=120,
            spike_offsets=(40, 80),
            spike_value=50.0,
        )
        mask = detect_nonwear_troiano(
            series, min_length="60min", spike_tolerance=2, spike_max_counts=100
        )
        assert masked_runs(mask) == [(300, 120)], "2 spikes are within the tolerance of 2"

    def test_exceeding_the_tolerance_splits_the_window_but_not_the_mask(self):
        """Exceeding ``spike_tolerance`` truncates the window; the scan restarts at the spike."""
        series = S.signal_with_nonwear(
            n_days=1,
            nonwear_start_epoch=300,
            nonwear_length_epochs=200,
            spike_offsets=(40, 80, 120),
            spike_value=50.0,
        )
        mask = detect_nonwear_troiano(
            series, min_length="60min", spike_tolerance=2, spike_max_counts=100
        )
        assert masked_runs(mask) == [(300, 200)]

    def test_tolerance_matters_when_the_remainder_is_too_short(self):
        """Breaking early on the third spike leaves a remainder too short to mask."""
        series = S.signal_with_nonwear(
            n_days=1,
            nonwear_start_epoch=300,
            nonwear_length_epochs=100,
            spike_offsets=(10, 20, 70),
            spike_value=50.0,
        )
        lenient = detect_nonwear_troiano(
            series, min_length="60min", spike_tolerance=3, spike_max_counts=100
        )
        strict = detect_nonwear_troiano(
            series, min_length="60min", spike_tolerance=2, spike_max_counts=100
        )
        assert masked_runs(lenient) == [(300, 100)]
        # Third spike breaks the window at 370; the 30-epoch remainder is too short
        assert masked_runs(strict) == [(300, 70)]

    def test_a_spike_above_spike_max_counts_is_genuine_activity(self):
        series = S.signal_with_nonwear(
            n_days=1,
            nonwear_start_epoch=300,
            nonwear_length_epochs=200,
            spike_offsets=(100,),
            spike_value=500.0,  # far above spike_max_counts
        )
        mask = detect_nonwear_troiano(
            series, min_length="60min", spike_tolerance=2, spike_max_counts=100
        )
        runs = masked_runs(mask)
        # Genuine activity splits the run into two halves of 100 each.
        assert runs == [(300, 100), (401, 99)]

    def test_spike_exactly_at_spike_max_counts_is_tolerated(self):
        """The rule is `counts <= spike_max_counts`, so 100 is a spike, not activity."""
        series = S.signal_with_nonwear(
            n_days=1,
            nonwear_start_epoch=300,
            nonwear_length_epochs=120,
            spike_offsets=(60,),
            spike_value=100.0,
        )
        mask = detect_nonwear_troiano(
            series, min_length="60min", spike_tolerance=2, spike_max_counts=100
        )
        assert masked_runs(mask) == [(300, 120)]

    def test_spike_one_above_the_threshold_is_activity(self):
        series = S.signal_with_nonwear(
            n_days=1,
            nonwear_start_epoch=300,
            nonwear_length_epochs=120,
            spike_offsets=(60,),
            spike_value=101.0,
        )
        mask = detect_nonwear_troiano(
            series, min_length="60min", spike_tolerance=2, spike_max_counts=100
        )
        # Activity at 360 splits the run; only the first 60 epochs reach the minimum
        assert masked_runs(mask) == [(300, 60)]


class TestTroianoBoundaries:
    def test_nonwear_at_the_very_start_is_detected(self):
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=0, nonwear_length_epochs=90
        )
        mask = detect_nonwear_troiano(series, min_length="60min", spike_tolerance=0)
        assert masked_runs(mask) == [(0, 90)]

    def test_nonwear_at_the_very_end_is_detected(self):
        n = 1440
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=n - 90, nonwear_length_epochs=90
        )
        mask = detect_nonwear_troiano(series, min_length="60min", spike_tolerance=0)
        assert masked_runs(mask) == [(n - 90, 90)]

    def test_entirely_zero_recording_is_entirely_non_wear(self):
        series = S.flat(n_days=1, value=0.0)
        mask = detect_nonwear_troiano(series, min_length="60min")
        assert (mask == 0).all(), "a dead device must be flagged as non-wear throughout"

    def test_recording_with_no_zeros_is_entirely_wear(self):
        series = S.flat(n_days=1, value=500.0)
        mask = detect_nonwear_troiano(series, min_length="60min")
        assert (mask == 1).all()

    def test_nan_input_does_not_crash(self):
        """NaN is neither zero nor above threshold; the mask must still be binary."""
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=300, nonwear_length_epochs=90
        )
        series.iloc[700:710] = np.nan
        mask = detect_nonwear_troiano(series, min_length="60min")
        A.assert_is_binary(mask)


class TestChoiNeighbourhoodRule:
    """Choi's addition over Troiano: the spike neighbourhood check."""

    def test_spike_deep_inside_a_long_run_is_tolerated(self):
        """Both 30 min neighbourhoods are inside the zero run, so the spike is fine."""
        series = S.signal_with_nonwear(
            n_days=1,
            nonwear_start_epoch=300,
            nonwear_length_epochs=200,
            spike_offsets=(100,),
            spike_value=50.0,
        )
        mask = detect_nonwear_choi(
            series, min_length="90min", window_size="30min", spike_tolerance=2
        )
        assert masked_runs(mask) == [(300, 200)]

    def test_spike_near_the_run_edge_fails_the_neighbourhood_check(self):
        """Choi refuses a spike with activity upstream; Troiano absorbs it."""
        series = S.signal_with_nonwear(
            n_days=1,
            nonwear_start_epoch=300,
            nonwear_length_epochs=200,
            spike_offsets=(5,),
            spike_value=50.0,
        )
        troiano = detect_nonwear_troiano(
            series, min_length="90min", spike_tolerance=2, spike_max_counts=100
        )
        choi = detect_nonwear_choi(
            series, min_length="90min", window_size="30min", spike_tolerance=2
        )

        # Troiano absorbs the spike and flags the whole run.
        assert masked_runs(troiano) == [(300, 200)]
        # Choi restarts after the spike, losing the first 6 epochs
        assert masked_runs(choi) == [(306, 194)]
        assert masked_runs(troiano) != masked_runs(choi)

    def test_choi_and_troiano_agree_when_there_are_no_spikes(self):
        """With a clean zero run the neighbourhood check is never invoked."""
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=300, nonwear_length_epochs=200
        )
        troiano = detect_nonwear_troiano(series, min_length="90min", spike_tolerance=0)
        choi = detect_nonwear_choi(series, min_length="90min", spike_tolerance=0)
        pd.testing.assert_series_equal(troiano, choi)

    def test_choi_default_min_length_is_ninety_minutes(self):
        """Choi et al. recommend 90 min; Troiano's NHANES default is 60."""
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=300, nonwear_length_epochs=75
        )
        assert masked_runs(detect_nonwear_troiano(series)) == [(300, 75)]
        assert masked_runs(detect_nonwear_choi(series)) == []

    def test_neighbourhood_check_only_fires_on_supra_threshold_activity(self):
        """Pin the *actual* neighbourhood predicate."""
        series = S.signal_with_nonwear(
            n_days=1,
            nonwear_start_epoch=300,
            nonwear_length_epochs=200,
            spike_offsets=(100,),
            spike_value=50.0,
        )
        # Sub-threshold epochs upstream: rejected by the spike counter, not the neighbourhood
        series.iloc[370:400] = 30.0
        choi = detect_nonwear_choi(
            series, min_length="90min", window_size="30min", spike_tolerance=2
        )
        assert (300, 200) not in masked_runs(choi)


class TestAlgorithmContract:
    """Properties that must hold for both algorithms on any input."""

    @pytest.mark.parametrize("detector", [detect_nonwear_troiano, detect_nonwear_choi])
    def test_output_is_binary_and_index_aligned(self, detector):
        series = S.realistic_rest_activity(n_days=3)
        mask = detector(series)
        A.assert_is_binary(mask)
        pd.testing.assert_index_equal(mask.index, series.index)
        assert len(mask) == len(series)

    @pytest.mark.parametrize("detector", [detect_nonwear_troiano, detect_nonwear_choi])
    def test_epochs_with_genuine_activity_are_never_non_wear(self, detector):
        series = S.signal_with_nonwear(
            n_days=2, nonwear_start_epoch=500, nonwear_length_epochs=150
        )
        mask = detector(series)
        high_activity = series > 100
        assert (mask[high_activity] == 1).all(), (
            "an epoch with counts above spike_max_counts can never be inside a "
            "non-wear window"
        )

    @pytest.mark.parametrize("detector", [detect_nonwear_troiano, detect_nonwear_choi])
    def test_longer_min_length_flags_no_more_than_shorter(self, detector):
        """Monotonicity: raising the length threshold can only shrink the mask."""
        series = S.realistic_rest_activity(n_days=3, sleep_duration_hours=9.0)
        short = detector(series, min_length="60min")
        long = detector(series, min_length="180min")
        assert (long == 0).sum() <= (short == 0).sum()

    @pytest.mark.parametrize("detector", [detect_nonwear_troiano, detect_nonwear_choi])
    def test_deterministic(self, detector):
        series = S.realistic_rest_activity(n_days=2)
        pd.testing.assert_series_equal(detector(series), detector(series))

    @pytest.mark.parametrize("sampling_period", [30, 60, 300])
    def test_result_is_consistent_across_epoch_lengths(self, sampling_period):
        """A 2 h non-wear block must be found whatever the epoch length."""
        epochs_per_2h = int(2 * 3600 / sampling_period)
        series = S.signal_with_nonwear(
            n_days=1,
            nonwear_start_epoch=epochs_per_2h,
            nonwear_length_epochs=epochs_per_2h,
            sampling_period=sampling_period,
        )
        mask = detect_nonwear_troiano(series, min_length="60min", spike_tolerance=0)
        assert masked_runs(mask) == [(epochs_per_2h, epochs_per_2h)]


class TestCrossImplementationAgreement:
    """The same algorithms are reachable through ``Mask.detect_nonwear``."""

    @staticmethod
    def _raw_from(series):
        """Wrap an activity series in a minimal ``Raw`` so ``Mask`` methods work."""
        from circstudio.io import Raw

        frequency = pd.Timedelta(S.DEFAULT_SAMPLING_PERIOD, unit="s")
        return Raw(
            df=pd.DataFrame({"activity": series}),
            period=frequency * len(series),
            frequency=frequency,
            activity=series,
            light=None,
            start_time=series.index[0],
        )

    @pytest.mark.parametrize("method", ["troiano", "choi"])
    def test_mask_api_matches_the_functional_api(self, method):
        series = S.signal_with_nonwear(
            n_days=2, nonwear_start_epoch=500, nonwear_length_epochs=150
        )
        direct = (
            detect_nonwear_troiano(series, min_length="90min")
            if method == "troiano"
            else detect_nonwear_choi(series, min_length="90min")
        )

        raw = self._raw_from(series)
        raw.detect_nonwear(method=method, min_length="90min")

        assert raw.mask is not None, "Mask.detect_nonwear did not store a mask"
        np.testing.assert_array_equal(np.asarray(raw.mask), np.asarray(direct))

    def test_unknown_method_is_rejected(self):
        raw = self._raw_from(S.signal_with_nonwear(n_days=1))
        with pytest.raises((ValueError, KeyError, NotImplementedError)):
            raw.detect_nonwear(method="not-a-real-algorithm")
