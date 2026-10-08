"""Checks that the synthetic signals and assertion helpers are themselves correct."""

import numpy as np
import pandas as pd
import pytest

from helpers import assertions as A
from helpers import signals as S


class TestSignalGenerators:
    def test_all_generators_return_regular_series(self):
        generators = {
            "sinewave": S.sinewave(n_days=2),
            "squarewave": S.squarewave(n_days=2),
            "two_tone": S.two_tone(n_days=2),
            "gaussian_noise": S.gaussian_noise(n_days=2),
            "brown_noise": S.brown_noise(n_days=2),
            "pink_noise": S.pink_noise(n_days=2),
            "realistic": S.realistic_rest_activity(n_days=2),
            "flat": S.flat(n_days=2),
            "single_spike": S.single_spike(n_days=1),
            "nonwear": S.signal_with_nonwear(n_days=2),
            "light": S.light_squarewave(n_days=2),
        }
        for name, series in generators.items():
            A.assert_valid_activity_series(series, name=name)

    def test_gap_generator_has_exactly_the_requested_nans(self):
        series = S.signal_with_gap(n_days=2, gap_start_epoch=100, gap_length_epochs=37)
        A.assert_valid_activity_series(series, name="gap", allow_nan=True)
        assert int(series.isna().sum()) == 37
        assert series.iloc[100:137].isna().all()
        assert series.iloc[:100].notna().all()
        assert series.iloc[137:].notna().all()

    @pytest.mark.parametrize("sampling_period", [15, 30, 60, 300])
    def test_epochs_per_day_is_exact(self, sampling_period):
        series = S.squarewave(n_days=3, sampling_period=sampling_period)
        assert len(series) == 3 * 86400 // sampling_period

    def test_sampling_period_that_does_not_divide_a_day_is_rejected(self):
        with pytest.raises(ValueError, match="does not divide"):
            S.squarewave(n_days=1, sampling_period=7)


class TestSquarewaveGroundTruth:
    """The square wave carries most of the suite's analytical ground truth."""

    def test_mean_matches_duty_cycle(self):
        series = S.squarewave(n_days=7, on_hours=12.0, low=0.0, high=100.0)
        assert series.mean() == pytest.approx(50.0)

        series = S.squarewave(n_days=7, on_hours=6.0, low=0.0, high=100.0)
        assert series.mean() == pytest.approx(25.0)

    def test_every_day_is_identical(self):
        """This is *why* IS must equal 1 for this signal."""
        series = S.squarewave(n_days=5)
        epd = 1440
        days = [series.values[i * epd : (i + 1) * epd] for i in range(5)]
        for day in days[1:]:
            np.testing.assert_array_equal(day, days[0])

    def test_transition_count_is_two_per_day(self):
        """The IV ground truth 8D/(N-1) depends on exactly 2 transitions/day."""
        n_days = 7
        series = S.squarewave(n_days=n_days)
        n_transitions = int((np.diff(series.values) != 0).sum())
        assert n_transitions == 2 * n_days - 1, (
            "the final off->on transition falls outside the series, so a "
            "D-day recording has 2D-1 interior transitions"
        )

    def test_iv_closed_form_is_self_consistent(self):
        """Verify the algebra in the docstring numerically, independent of circStudio."""
        n_days, high = 7, 100.0
        series = S.squarewave(n_days=n_days, on_hours=12.0, low=0.0, high=high)
        x = series.values
        n = len(x)
        numerator = n * np.sum(np.diff(x) ** 2)
        denominator = (n - 1) * np.sum((x - x.mean()) ** 2)
        iv = numerator / denominator
        # Closed form: 8D/(N-1), adjusted for the 2D-1 interior transitions.
        expected = n * (2 * n_days - 1) * high**2 / ((n - 1) * n * high**2 / 4)
        assert iv == pytest.approx(expected, rel=1e-12)

    def test_l5_m10_windows_are_pure(self):
        """L5 sits wholly in the off block and M10 wholly in the on block."""
        series = S.squarewave(n_days=7, on_hours=12.0, low=0.0, high=100.0)
        rolling_5h = series.rolling(window=300, center=False).mean()
        rolling_10h = series.rolling(window=600, center=False).mean()
        assert rolling_5h.min() == pytest.approx(0.0)
        assert rolling_10h.max() == pytest.approx(100.0)


class TestSinewaveGroundTruth:
    def test_mean_over_whole_periods_is_the_mesor(self):
        series = S.sinewave(n_days=7, mesor=250.0, amplitude=100.0)
        assert series.mean() == pytest.approx(250.0, abs=1e-9)

    def test_maximum_occurs_at_the_acrophase(self):
        series = S.sinewave(n_days=1, acrophase_hours=6.0, amplitude=100.0, mesor=0.0)
        peak_time = series.idxmax()
        assert peak_time.hour == 6 and peak_time.minute == 0

    def test_amplitude_is_half_the_peak_to_trough_range(self):
        series = S.sinewave(n_days=2, amplitude=42.0, mesor=10.0)
        assert (series.max() - series.min()) / 2 == pytest.approx(42.0, rel=1e-9)

    def test_noise_is_reproducible_for_a_given_seed(self):
        a = S.sinewave(n_days=1, noise_sd=5.0, seed=123)
        b = S.sinewave(n_days=1, noise_sd=5.0, seed=123)
        c = S.sinewave(n_days=1, noise_sd=5.0, seed=124)
        pd.testing.assert_series_equal(a, b)
        assert not a.equals(c)


class TestRestActivityGroundTruth:
    def test_sleep_window_is_where_it_was_asked_for(self):
        series = S.realistic_rest_activity(
            n_days=3, sleep_start_hour=23.0, sleep_duration_hours=8.0, n_awakenings=0
        )
        # Night 1: 23:00 on day 0 to 07:00 on day 1.
        night = series["2020-01-01 23:00":"2020-01-02 06:59"]
        assert (night == 0).all(), "sleep window should be all zeros with no awakenings"
        # Immediately before sleep onset the subject is active.
        assert series["2020-01-01 22:00":"2020-01-01 22:59"].sum() > 0

    def test_awakenings_split_the_night_into_known_bout_count(self):
        n_awakenings = 3
        series = S.realistic_rest_activity(
            n_days=1, sleep_start_hour=1.0, sleep_duration_hours=8.0, n_awakenings=n_awakenings
        )
        night = series["2020-01-01 01:00":"2020-01-01 08:59"]
        zero_runs = A.count_runs(night.values == 0)
        assert len(zero_runs) == n_awakenings + 1

    def test_activity_is_non_negative(self):
        series = S.realistic_rest_activity(n_days=3)
        assert (series >= 0).all()


class TestNonwearGenerator:
    def test_zero_run_is_exactly_where_requested(self):
        series = S.signal_with_nonwear(
            n_days=1, nonwear_start_epoch=100, nonwear_length_epochs=90
        )
        runs = A.count_runs(series.values == 0)
        assert runs == [(100, 90)]

    def test_spikes_break_the_run_into_known_pieces(self):
        series = S.signal_with_nonwear(
            n_days=1,
            nonwear_start_epoch=100,
            nonwear_length_epochs=90,
            spike_offsets=(30,),
            spike_value=50.0,
        )
        runs = A.count_runs(series.values == 0)
        assert runs == [(100, 30), (131, 59)]
        assert series.iloc[130] == 50.0


class TestAssertionHelpers:
    def test_count_runs_on_hand_written_arrays(self):
        assert A.count_runs([0, 0, 0]) == []
        assert A.count_runs([1, 1, 1]) == [(0, 3)]
        assert A.count_runs([1, 0, 1]) == [(0, 1), (2, 1)]
        assert A.count_runs([0, 1, 1, 0, 0, 1]) == [(1, 2), (5, 1)]
        assert A.count_runs([]) == []

    def test_assert_within_range_catches_violations(self):
        A.assert_within_range([0.0, 0.5, 1.0], 0.0, 1.0)
        with pytest.raises(AssertionError, match="outside"):
            A.assert_within_range([0.0, 1.5], 0.0, 1.0)

    def test_assert_within_range_ignores_non_finite(self):
        A.assert_within_range([0.5, np.nan, np.inf], 0.0, 1.0)

    def test_assert_datetime_index_regular_rejects_irregular(self):
        irregular = pd.Series(
            [1, 2, 3],
            index=pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:01", "2020-01-01 00:05"]),
        )
        with pytest.raises(AssertionError, match="irregular"):
            A.assert_datetime_index_regular(irregular)

    def test_assert_is_binary(self):
        A.assert_is_binary([0, 1, 1, 0])
        with pytest.raises(AssertionError, match="outside"):
            A.assert_is_binary([0, 1, 2])

    def test_assert_runs_equal_message_is_informative(self):
        with pytest.raises(AssertionError, match=r"runs \[\(0, 1\)\]"):
            A.assert_runs_equal([1, 0, 0], [(0, 2)])

    def test_assert_figure_renders_accepts_matplotlib_and_plotly(self):
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        A.assert_figure_renders(fig)
        plt.close(fig)

        pfig = go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])])
        A.assert_figure_renders(pfig)

    def test_assert_figure_renders_rejects_empty(self):
        import plotly.graph_objects as go

        with pytest.raises(AssertionError, match="traces"):
            A.assert_figure_renders(go.Figure())


class TestConftestWiring:
    def test_circstudio_is_importable_under_the_agreed_convention(self):
        import circstudio

        assert hasattr(circstudio, "io")
        assert hasattr(circstudio, "analysis")
        assert hasattr(circstudio, "preprocessing")

    def test_src_prefixed_imports_are_not_used(self):
        """Guard against regressing to the old ``from src import circstudio`` style."""
        import sys

        assert "src.circstudio" not in sys.modules

    def test_matplotlib_is_headless(self):
        import matplotlib

        assert matplotlib.get_backend().lower() == "agg"

    def test_synthetic_raw_fixture_builds(self, synthetic_raw):
        A.assert_valid_activity_series(synthetic_raw.activity)
        assert synthetic_raw.light is not None
