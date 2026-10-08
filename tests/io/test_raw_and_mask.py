"""The ``Raw`` container, ``BaseLog`` and ``Mask``."""

import numpy as np
import pandas as pd
import pytest

from circstudio.io import Raw, read_awd
from circstudio.io.mask import BaseLog
from helpers import assertions as A
from helpers import signals as S

ONE_MINUTE = pd.Timedelta(60, unit="s")


def build_raw(activity, light=None):
    """Wrap a series in a minimal ``Raw``."""
    frame = {"activity": activity}
    if light is not None:
        frame["light"] = light
    return Raw(
        df=pd.DataFrame(frame),
        period=ONE_MINUTE * len(activity),
        frequency=ONE_MINUTE,
        activity=activity,
        light=light,
        start_time=activity.index[0],
    )


# Raw


class TestRawConstruction:
    def test_builds_from_a_synthetic_series(self):
        raw = build_raw(S.squarewave(n_days=3))
        A.assert_valid_activity_series(raw.activity)
        assert raw.frequency == ONE_MINUTE

    def test_length_counts_epochs(self):
        activity = S.squarewave(n_days=3)
        assert build_raw(activity).length() == len(activity)

    def test_duration_is_length_times_frequency(self):
        """``duration()`` counts epochs; it is one epoch longer than the span."""
        activity = S.squarewave(n_days=3)
        raw = build_raw(activity)
        assert raw.duration() == len(activity) * ONE_MINUTE

    def test_time_range_is_the_span_between_first_and_last_epoch(self):
        activity = S.squarewave(n_days=3)
        raw = build_raw(activity)
        assert raw.time_range() == (len(activity) - 1) * ONE_MINUTE

    def test_duration_and_time_range_differ_by_exactly_one_epoch(self):
        """A recurring source of off-by-one errors; pin the relationship."""
        raw = build_raw(S.squarewave(n_days=3))
        assert raw.duration() - raw.time_range() == raw.frequency

    def test_light_is_optional(self):
        assert build_raw(S.squarewave(n_days=2)).light is None

    def test_light_when_supplied_is_aligned(self):
        activity = S.squarewave(n_days=2)
        raw = build_raw(activity, light=S.light_squarewave(n_days=2))
        assert raw.light is not None
        pd.testing.assert_index_equal(raw.light.index, raw.activity.index)


@pytest.mark.needs_data
class TestRawFromFile:
    def test_bundled_recording_is_internally_consistent(self, raw_awd):
        assert raw_awd.length() == len(raw_awd.activity)
        assert raw_awd.duration() == raw_awd.length() * raw_awd.frequency
        assert raw_awd.time_range() == raw_awd.duration() - raw_awd.frequency

    def test_plot_returns_a_figure(self, raw_awd):
        A.assert_figure_renders(raw_awd.plot(mode="activity"))

    def test_plot_accepts_a_log_scale(self, raw_awd):
        A.assert_figure_renders(raw_awd.plot(mode="activity", log=True))

    def test_plotting_a_missing_channel_fails_clearly(self, raw_awd):
        """``example_01.AWD`` has no light channel."""
        assert raw_awd.light is None
        with pytest.raises(Exception):
            raw_awd.plot(mode="light")


# BaseLog


@pytest.mark.needs_data
class TestBaseLog:
    def test_reads_a_csv_log(self, mask_log_csv):
        result = BaseLog.from_file(str(mask_log_csv), "Subject_id")
        assert result is not None

    def test_every_container_format_gives_the_same_log(self, sst_logs):
        """The same start/stop-time log ships as .csv, .ods, .xls and .xlsx."""
        if len(sst_logs) < 2:
            pytest.skip("fewer than two container formats available")

        parsed = {}
        for fmt, path in sst_logs.items():
            try:
                parsed[fmt] = BaseLog.from_file(str(path), "Subject_id")
            except Exception as error:  # noqa: BLE001 - reported below
                pytest.skip(f"{fmt} could not be parsed: {type(error).__name__}: {error}")

        frames = {}
        for fmt, result in parsed.items():
            log = result[1] if isinstance(result, tuple) else getattr(result, "log", result)
            frames[fmt] = log

        reference_fmt, reference = next(iter(frames.items()))
        for fmt, frame in frames.items():
            if fmt == reference_fmt:
                continue
            assert len(frame) == len(reference), (
                f"{fmt} parsed {len(frame)} rows but {reference_fmt} parsed "
                f"{len(reference)}"
            )

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            BaseLog.from_file(str(tmp_path / "absent.csv"), "Subject_id")

    def test_missing_index_column_raises(self, tmp_path):
        path = tmp_path / "log.csv"
        path.write_text("A,B\n1,2\n")
        with pytest.raises(Exception):
            BaseLog.from_file(str(path), "Subject_id")


# Mask -- filtering


class TestFiltering:
    @pytest.fixture
    def raw(self):
        return build_raw(S.realistic_rest_activity(n_days=3))

    def test_binarising_produces_only_zeros_and_ones(self, raw):
        raw.apply_filters(binarize=True, threshold=10)
        A.assert_is_binary(raw.activity)

    def test_binarising_preserves_the_epoch_count(self, raw):
        before = len(raw.activity)
        raw.apply_filters(binarize=True, threshold=10)
        assert len(raw.activity) == before

    def test_binarising_uses_a_strict_threshold(self):
        raw = build_raw(S.as_series([0.0, 5.0, 10.0, 20.0] * 360))
        raw.apply_filters(binarize=True, threshold=10)
        np.testing.assert_array_equal(raw.activity.values[:4], [0, 0, 0, 1])

    def test_reset_restores_the_original_exactly(self, raw):
        original = raw.activity.copy()
        raw.apply_filters(binarize=True, threshold=10)
        assert not raw.activity.equals(original)
        raw.reset_filters()
        pd.testing.assert_series_equal(raw.activity, original)

    def test_applying_the_same_filter_twice_is_idempotent(self, raw):
        raw.apply_filters(binarize=True, threshold=10)
        once = raw.activity.copy()
        raw.apply_filters(binarize=True, threshold=10)
        pd.testing.assert_series_equal(raw.activity, once)

    def test_resampling_reduces_the_epoch_count(self, raw):
        before = len(raw.activity)
        raw.apply_filters(new_freq="10min")
        assert len(raw.activity) < before

    def test_resampling_conserves_total_activity(self, raw):
        total = raw.activity.sum()
        raw.apply_filters(new_freq="10min")
        assert raw.activity.sum() == pytest.approx(total, rel=1e-9)

    def test_default_filtering_is_a_no_op(self, raw):
        original = raw.activity.copy()
        raw.apply_filters()
        pd.testing.assert_series_equal(raw.activity, original)

    @pytest.mark.parametrize("method", ["mean", "median"])
    def test_imputation_fills_every_gap(self, method):
        raw = build_raw(S.signal_with_gap(n_days=3, gap_start_epoch=100, gap_length_epochs=30))
        assert raw.activity.isna().sum() == 30
        raw.apply_filters(impute_nan=True, imputation_method=method)
        assert raw.activity.isna().sum() == 0


# Mask -- exclusion


class TestInactivityMask:
    @pytest.fixture
    def raw(self):
        return build_raw(
            S.signal_with_nonwear(
                n_days=2, nonwear_start_epoch=500, nonwear_length_epochs=150
            )
        )

    def test_no_mask_exists_by_default(self, raw):
        assert raw.mask is None

    def test_create_inactivity_mask_flags_the_zero_run(self, raw):
        raw.create_inactivity_mask("60min")
        assert raw.mask is not None
        assert A.count_runs(raw.mask.values == 0) == [(500, 150)]

    def test_a_run_shorter_than_the_threshold_is_left_alone(self):
        raw = build_raw(
            S.signal_with_nonwear(n_days=2, nonwear_start_epoch=500, nonwear_length_epochs=30)
        )
        raw.create_inactivity_mask("60min")
        assert (raw.mask == 1).all()

    def test_mask_is_aligned_with_the_recording(self, raw):
        raw.create_inactivity_mask("60min")
        assert len(raw.mask) == len(raw.activity)
        pd.testing.assert_index_equal(raw.mask.index, raw.activity.index)

    def test_add_mask_period_marks_the_requested_interval(self, raw):
        start = raw.activity.index[100]
        stop = raw.activity.index[200]
        raw.add_mask_period(start, stop)
        assert raw.mask is not None
        masked = raw.mask[raw.mask == 0]
        assert masked.index.min() >= start
        assert masked.index.max() <= stop

    def test_epochs_outside_an_added_period_stay_unmasked(self, raw):
        raw.add_mask_period(raw.activity.index[100], raw.activity.index[200])
        assert raw.mask.iloc[0] == 1
        assert raw.mask.iloc[-1] == 1

    def test_two_added_periods_are_both_masked(self, raw):
        raw.add_mask_period(raw.activity.index[100], raw.activity.index[150])
        raw.add_mask_period(raw.activity.index[400], raw.activity.index[450])
        runs = A.count_runs(raw.mask.values == 0)
        assert len(runs) == 2

    def test_masking_changes_the_filtered_signal(self, raw):
        raw.create_inactivity_mask("60min")
        unmasked_total = raw.activity.sum()
        raw.apply_filters(apply_mask=True)
        assert raw.activity.sum() <= unmasked_total


class TestMaskDetectNonwear:
    """``Mask.detect_nonwear`` is the object-level entry to the algorithms."""

    @pytest.fixture
    def raw(self):
        return build_raw(
            S.signal_with_nonwear(
                n_days=2, nonwear_start_epoch=500, nonwear_length_epochs=150
            )
        )

    @pytest.mark.parametrize("method", ["choi", "troiano"])
    def test_stores_a_mask_matching_the_functional_api(self, raw, method):
        from circstudio.preprocessing import detect_nonwear_choi, detect_nonwear_troiano

        expected = (
            detect_nonwear_choi(raw.activity, min_length="90min")
            if method == "choi"
            else detect_nonwear_troiano(raw.activity, min_length="90min")
        )
        raw.detect_nonwear(method=method, min_length="90min")
        np.testing.assert_array_equal(np.asarray(raw.mask), np.asarray(expected))

    def test_detected_mask_is_binary(self, raw):
        raw.detect_nonwear(method="choi")
        A.assert_is_binary(raw.mask)

    def test_unknown_method_is_rejected(self, raw):
        with pytest.raises((ValueError, KeyError, NotImplementedError)):
            raw.detect_nonwear(method="not-an-algorithm")

    def test_detection_replaces_any_previous_mask(self, raw):
        """Documented behaviour: a previously set inactivity mask is discarded."""
        raw.add_mask_period(raw.activity.index[10], raw.activity.index[20])
        before = A.count_runs(raw.mask.values == 0)
        raw.detect_nonwear(method="troiano", min_length="90min")
        after = A.count_runs(raw.mask.values == 0)
        assert after != before
