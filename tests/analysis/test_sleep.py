"""Sleep scoring, sleep summary measures and the sleep diary."""

import numpy as np
import pandas as pd
import pytest

from circstudio.analysis import (
    CSM,
    AoffT,
    AonT,
    Cole_Kripke,
    Crespo,
    Oakley,
    Roenneberg,
    Roenneberg_AoT,
    Sadeh,
    Scripps,
    SleepDiary,
    SleepMidPoint,
    SleepProfile,
    SleepRegularityIndex,
    active_bouts,
    active_durations,
    fSoD,
    main_sleep_bouts,
    sleep_bouts,
    sleep_durations,
    waso,
)
from circstudio.analysis.sleep.scoring.sri import prob_stability, sri, sri_profile
from circstudio.analysis.sleep.scoring.utils import (
    consecutive_values,
    rescore,
    rescore_if_preceded,
    rescore_if_surrounded,
    rolling_window,
)
from helpers import assertions as A
from helpers import signals as S

# Scorers that take a plain activity series and return a binary Series.
BINARY_SCORERS = [Sadeh, Scripps, Oakley, Roenneberg]
ABSOLUTE_SCORERS = [Sadeh, Scripps, Oakley]  # threshold against counts, not a trend

ONE_MINUTE = pd.Timedelta(60, unit="s")


@pytest.fixture(scope="module")
def night_signal():
    """3 nights, sleep 23:00-07:00, no awakenings -- the cleanest test case."""
    return S.realistic_rest_activity(
        n_days=3, sleep_start_hour=23, sleep_duration_hours=8, n_awakenings=0
    )


@pytest.fixture(scope="module")
def broken_night_signal():
    """3 nights with two 10-minute awakenings each."""
    return S.realistic_rest_activity(
        n_days=3, sleep_start_hour=23, sleep_duration_hours=8, n_awakenings=2
    )


def night_mask(series):
    """Boolean mask of the 23:00-07:00 window."""
    return (series.index.hour >= 23) | (series.index.hour < 7)


# Webster rescoring rules


class TestWebsterRescoringRules:
    """Carried over and extended from the legacy ``tests/test_ck.py``."""

    LEGACY_VECTOR = np.asarray(
        [0] * 4 + [1]
        + [0] * 10 + [1, 1, 1]
        + [0] * 15 + [1, 1, 1, 1]
        + [0] * 10 + [1] * 6 + [0] * 10
        + [0] * 20 + [1] * 10 + [0] * 20
    )

    def test_every_sleep_epoch_in_the_legacy_vector_is_rescored(self):
        """The vector is built so that every sleep epoch violates a rule."""
        result = rescore(self.LEGACY_VECTOR, sleep_score=1)
        expected = np.logical_not(self.LEGACY_VECTOR).astype(float)
        np.testing.assert_array_equal(result, expected)

    def test_wake_epochs_are_never_rescored(self):
        result = rescore(self.LEGACY_VECTOR, sleep_score=1)
        assert (result[self.LEGACY_VECTOR == 0] == 1).all()

    def test_rule_one_clips_an_isolated_minute_after_four_of_wake(self):
        series = np.asarray([0] * 4 + [1] + [0] * 30)
        assert rescore(series, sleep_score=1)[4] == 0

    def test_three_minutes_of_wake_is_not_enough_for_rule_one(self):
        """The threshold is 4 minutes; 3 must leave the sleep epoch alone."""
        series = np.asarray([1] * 20 + [0] * 3 + [1] * 21)
        assert rescore(series, sleep_score=1)[23] == 1

    def test_a_long_sleep_block_mostly_survives(self):
        """Rules 4 and 5 only discard blocks of <= 10 min."""
        series = np.asarray([0] * 20 + [1] * 120 + [0] * 20)
        result = rescore(series, sleep_score=1)
        assert (result[20:24] == 0).all(), "rule 3 clips the first 4 minutes"
        assert (result[24:140] == 1).all(), "the bulk of a 2 h block must survive"

    def test_rescoring_never_creates_sleep(self):
        """The rules only ever convert sleep to wake, never the reverse."""
        rng = np.random.default_rng(0)
        series = rng.integers(0, 2, 500)
        result = rescore(series, sleep_score=1)
        assert not ((series == 0) & (result == 0)).any()

    def test_preceded_and_surrounded_rules_differ(self):
        """``n_periods``: sleep-block length; ``n_previous``/``n_surround``: wake required around it."""
        series = np.asarray([0] * 20 + [1] * 5 + [0] * 3 + [1] * 40)
        preceded = rescore_if_preceded(series, n_periods=5, n_previous=10, sleep_score=1)
        surrounded = rescore_if_surrounded(series, n_periods=5, n_surround=10, sleep_score=1)
        assert not np.array_equal(preceded, surrounded), (
            "a block with wake before but not after must be treated differently"
        )


class TestScoringUtilities:
    def test_rolling_window_shape_and_contents(self):
        windows = rolling_window(np.arange(10), 3)
        assert windows.shape == (8, 3)
        np.testing.assert_array_equal(windows[0], [0, 1, 2])
        np.testing.assert_array_equal(windows[-1], [7, 8, 9])

    def test_consecutive_values_honours_min_length(self):
        series = np.asarray([1] * 12 + [0] * 5 + [1] * 3)
        assert np.asarray(consecutive_values(series, target=1, min_length=10)).any()
        assert not np.asarray(consecutive_values(series, target=1, min_length=20)).any()


# Scoring algorithms


class TestScoringConvention:
    def test_one_means_sleep(self, night_signal):
        """Establish the polarity rather than assuming it."""
        scored = Roenneberg(night_signal)
        mask = night_mask(night_signal)
        assert scored[mask].mean() > 0.5, "night epochs must mostly score 1"
        assert scored[~mask].mean() < 0.1, "daytime epochs must mostly score 0"

    @pytest.mark.parametrize("scorer", BINARY_SCORERS, ids=lambda f: f.__name__)
    def test_output_is_binary_and_aligned(self, scorer, night_signal):
        scored = scorer(night_signal)
        assert isinstance(scored, pd.Series)
        A.assert_is_binary(scored)
        pd.testing.assert_index_equal(scored.index, night_signal.index)


class TestScoringAlgorithms:
    @pytest.mark.parametrize("scorer", BINARY_SCORERS, ids=lambda f: f.__name__)
    def test_all_algorithms_find_the_known_sleep_window(self, scorer, night_signal):
        """Algorithms disagree in detail; none may score the daytime as sleep."""
        scored = scorer(night_signal)
        mask = night_mask(night_signal)
        assert scored[mask].mean() > 0.5
        assert scored[~mask].mean() < 0.2

    @pytest.mark.parametrize("scorer", ABSOLUTE_SCORERS, ids=lambda f: f.__name__)
    def test_a_permanently_active_recording_scores_no_sleep(self, scorer):
        awake = S.gaussian_noise(n_days=2, mu=500.0, sigma=50.0)
        assert scorer(awake).mean() < 0.2

    def test_oakley_threshold_is_monotonic(self, night_signal):
        fractions = [Oakley(night_signal, threshold=t).mean() for t in (10, 40, 80, 160)]
        A.assert_monotonic(fractions, increasing=True)

    @pytest.mark.parametrize("threshold", [0.0, 1.0, 5.0])
    def test_scripps_threshold_stays_in_range(self, threshold, night_signal):
        A.assert_within_range(
            Scripps(night_signal, threshold=threshold).mean(), 0.0, 1.0, name="Scripps"
        )

    def test_published_constants_are_the_defaults(self):
        """A silent typo in a published coefficient is invisible; check them."""
        import inspect

        assert inspect.signature(Sadeh).parameters["offset"].default == 7.601
        assert inspect.signature(Scripps).parameters["scale"].default == 0.204
        assert inspect.signature(Oakley).parameters["threshold"].default == 40

    def test_activity_onset_and_offset_are_timedeltas_half_a_day_apart(
        self, night_signal
    ):
        onset, offset = AonT(night_signal), AoffT(night_signal)
        assert isinstance(onset, pd.Timedelta)
        assert isinstance(offset, pd.Timedelta)
        gap = abs((offset - onset).total_seconds()) / 3600
        assert 6 < gap < 18

    def test_roenneberg_aot_returns_onsets_and_offsets(self, night_signal):
        onsets, offsets = Roenneberg_AoT(night_signal)
        assert len(onsets) > 0 and len(offsets) > 0

    def test_csm_returns_a_series(self, night_signal):
        assert isinstance(CSM(night_signal), pd.Series)

    def test_fsod_is_a_scalar(self, night_signal):
        assert np.isfinite(float(fSoD(night_signal)))


class TestRoennebergRobustness:
    """Roenneberg thresholds against a running trend, not absolute counts."""

    def test_raises_when_no_sleep_bout_seed_survives(self):
        """It crashes rather than reporting "no sleep found"."""
        with pytest.raises(IndexError):
            Roenneberg(S.gaussian_noise(n_days=2, mu=500.0, sigma=50.0))

    def test_a_threshold_too_low_to_seed_a_bout_also_crashes(self):
        """The same empty-seed failure, reached via the threshold instead."""
        gradual = S.sinewave(n_days=4, mesor=100.0, amplitude=95.0, acrophase_hours=14)
        with pytest.raises(IndexError):
            Roenneberg(gradual, threshold=0.05)

    def test_threshold_is_inert_on_a_sharp_transition(self, night_signal):
        """On a square rest-activity profile the threshold does not matter."""
        fractions = [Roenneberg(night_signal, threshold=t).mean() for t in (0.05, 0.5, 0.9)]
        assert len(set(np.round(fractions, 10))) == 1

    def test_threshold_matters_on_a_gradual_signal(self):
        """Contrast with the square-wave case: 0.14 at 0.15 rising to 0.43 at 0.90."""
        gradual = S.sinewave(n_days=4, mesor=100.0, amplitude=95.0, acrophase_hours=14)
        fractions = [Roenneberg(gradual, threshold=t).mean() for t in (0.15, 0.5, 0.9)]
        A.assert_monotonic(fractions, increasing=True, strict=True)


class TestCrespo:
    def test_requires_a_timedelta_frequency(self, night_signal):
        """Passing ``series.index.freq`` -- the obvious thing -- does not work."""
        with pytest.raises(AttributeError, match="total_seconds"):
            Crespo(night_signal, frequency=night_signal.index.freq)

    def test_runs_with_an_explicit_timedelta(self, night_signal):
        assert len(Crespo(night_signal, frequency=ONE_MINUTE)) > 0


class TestColeKripke:
    """Cole-Kripke needs sub-minute epochs under all but one setting."""

    def test_works_at_thirty_second_epochs(self):
        scored = Cole_Kripke(S.realistic_rest_activity(n_days=2, sampling_period=30))
        A.assert_is_binary(scored)

    def test_mean_setting_works_at_one_minute_epochs(self, night_signal):
        A.assert_is_binary(Cole_Kripke(night_signal, settings="mean"))

    def test_rescoring_only_removes_sleep(self):
        series = S.realistic_rest_activity(n_days=2, sampling_period=30)
        plain = Cole_Kripke(series, rescoring=False)
        rescored = Cole_Kripke(series, rescoring=True)
        assert rescored.sum() <= plain.sum()

    def test_unsupported_setting_lists_the_alternatives(self, night_signal):
        with pytest.raises(ValueError, match="(?i)available settings"):
            Cole_Kripke(night_signal, settings="60sec_max_non_overlap")

    @pytest.mark.xfail(reason="Cole-Kripke's default setting rejects 1-minute data", strict=True)
    def test_default_settings_accept_one_minute_data(self, night_signal):
        Cole_Kripke(night_signal)

    def test_the_rejection_is_pinned(self, night_signal):
        with pytest.raises(ValueError, match="sampling frequency"):
            Cole_Kripke(night_signal)


# Bouts and durations


class TestBouts:
    def test_uninterrupted_nights_give_one_bout_each(self, night_signal):
        """3 nights, but the last is truncated by the end of the recording."""
        bouts = sleep_bouts(night_signal)
        assert len(bouts) == 2
        for bout in bouts:
            A.assert_timedelta_close(
                bout.index[-1] - bout.index[0],
                pd.Timedelta(8, unit="h"),
                pd.Timedelta(15, unit="m"),
            )

    def test_bouts_are_series_and_durations_are_timedeltas(self, night_signal):
        assert all(isinstance(b, pd.Series) for b in sleep_bouts(night_signal))
        assert all(isinstance(d, pd.Timedelta) for d in sleep_durations(night_signal))

    def test_durations_match_the_bouts(self, night_signal):
        bouts = sleep_bouts(night_signal)
        durations = sleep_durations(night_signal)
        assert len(bouts) == len(durations)
        for bout, duration in zip(bouts, durations):
            assert duration == bout.index[-1] - bout.index[0]

    def test_short_awakenings_are_absorbed_by_bout_cleaning(
        self, night_signal, broken_night_signal
    ):
        """Roenneberg deliberately merges brief awakenings back into the bout."""
        assert len(sleep_bouts(broken_night_signal)) == len(sleep_bouts(night_signal))

    def test_a_long_awakening_still_costs_sleep_time(self):
        clean = S.realistic_rest_activity(
            n_days=3, sleep_start_hour=23, sleep_duration_hours=8, n_awakenings=0
        )
        broken = S.realistic_rest_activity(
            n_days=3,
            sleep_start_hour=23,
            sleep_duration_hours=8,
            n_awakenings=1,
            awakening_minutes=90,
        )
        assert sum(sleep_durations(broken), pd.Timedelta(0)) <= sum(
            sleep_durations(clean), pd.Timedelta(0)
        )

    def test_bouts_share_their_boundary_epoch(self, night_signal):
        """Bouts are *closed* intervals, so each transition epoch belongs to two."""
        sleep_span = set()
        for bout in sleep_bouts(night_signal):
            sleep_span.update(bout.index)
        active_span = set()
        for bout in active_bouts(night_signal):
            active_span.update(bout.index)

        shared = sleep_span & active_span
        n_transitions = len(sleep_bouts(night_signal)) + len(active_bouts(night_signal)) - 1
        assert len(shared) == n_transitions, (
            "exactly one shared epoch per transition -- an inclusive-endpoint "
            "off-by-one, not an arbitrary overlap"
        )

    def test_duration_filters_exclude_out_of_range_bouts(self, night_signal):
        assert len(sleep_bouts(night_signal, duration_min="10h", duration_max="24h")) == 0
        assert len(sleep_bouts(night_signal, duration_min="1min", duration_max="24h")) == 2

    def test_active_durations_are_all_positive(self, night_signal):
        assert all(d > pd.Timedelta(0) for d in active_durations(night_signal))

    def test_main_sleep_bouts_returns_a_table_and_a_total(self, night_signal):
        table, total = main_sleep_bouts(night_signal)
        assert isinstance(table, pd.DataFrame)
        assert list(table.columns) == ["start_time", "stop_time", "duration", "date"]
        assert isinstance(total, pd.Timedelta)
        assert (table["duration"] > pd.Timedelta(0)).all()
        assert (table["stop_time"] > table["start_time"]).all()

    def test_algo_argument_accepts_roenneberg(self, night_signal):
        assert isinstance(sleep_bouts(night_signal, algo="Roenneberg"), list)

    @pytest.mark.parametrize("algo", ["Sadeh", "Scripps", "Oakley"])
    def test_scorers_without_an_aot_variant_fail_with_a_raw_keyerror(
        self, night_signal, algo
    ):
        """The failure names an internal symbol rather than the real problem."""
        with pytest.raises(KeyError, match=f"{algo}_AoT"):
            sleep_bouts(night_signal, algo=algo)


# Summary measures


class TestSleepProfile:
    def test_values_are_probabilities_over_a_day(self, night_signal):
        profile = SleepProfile(night_signal)
        assert isinstance(profile.index, pd.TimedeltaIndex)
        A.assert_within_range(profile.values, 0.0, 1.0, name="sleep profile")

    @pytest.mark.parametrize("freq,expected", [("15min", 96), ("30min", 48)])
    def test_resampling_frequency_sets_the_resolution(self, night_signal, freq, expected):
        assert len(SleepProfile(night_signal, freq=freq)) == expected

    def test_profile_peaks_during_the_known_sleep_window(self, night_signal):
        peak_hour = SleepProfile(night_signal, freq="30min").idxmax().total_seconds() / 3600
        assert peak_hour >= 23 or peak_hour < 7


class TestSleepRegularityIndex:
    def test_a_perfectly_repeating_recording_scores_one_hundred(self):
        assert sri(S.flat(n_days=5, value=0.0)) == pytest.approx(100.0)

    def test_index_lies_in_the_documented_range(self, broken_night_signal):
        A.assert_within_range(
            SleepRegularityIndex(broken_night_signal), -100.0, 100.0, name="SRI"
        )

    def test_a_realistic_schedule_scores_below_a_perfectly_flat_one(self):
        perfect = sri(S.flat(n_days=5, value=0.0))
        realistic = SleepRegularityIndex(
            S.realistic_rest_activity(n_days=6, n_awakenings=0, seed=1)
        )
        assert perfect == pytest.approx(100.0)
        assert realistic < perfect

    def test_index_inherits_the_roenneberg_crash(self):
        """SRI defaults to algo='Roenneberg' and inherits its empty-seed crash."""
        with pytest.raises(IndexError):
            SleepRegularityIndex(S.gaussian_noise(n_days=6, mu=200.0, sigma=200.0, seed=1))

    def test_probability_of_stability_is_a_probability(self, night_signal):
        A.assert_within_range(
            prob_stability(night_signal, threshold=0.15), 0.0, 1.0, name="prob_stability"
        )

    def test_profile_values_are_probabilities(self, night_signal):
        profile = sri_profile(night_signal, threshold=0.15)
        assert len(profile) > 0
        A.assert_within_range(profile, 0.0, 1.0, name="SRI profile")


class TestSleepMidPoint:
    def test_returns_a_timedelta_inside_the_night(self, night_signal):
        midpoint = SleepMidPoint(night_signal)
        assert isinstance(midpoint, pd.Timedelta)
        hour = midpoint.total_seconds() / 3600 % 24
        assert 1.0 <= hour <= 5.0, "sleep runs 23:00-07:00, so the midpoint is ~03:00"

    def test_handles_the_midnight_crossing(self, night_signal):
        """A 23:00-07:00 sleep period has its midpoint at 03:00, not 15:00."""
        hour = SleepMidPoint(night_signal).total_seconds() / 3600 % 24
        assert abs(hour - 15.0) > 6.0, "must not return the naive arithmetic mean"

    def test_to_td_false_returns_a_finite_number(self, night_signal):
        as_num = SleepMidPoint(night_signal, to_td=False)
        assert not isinstance(as_num, pd.Timedelta)
        assert np.isfinite(float(as_num))


class TestWaso:
    def test_returns_per_day_values_and_a_total(self, broken_night_signal):
        per_day, total = waso(broken_night_signal, frequency=ONE_MINUTE)
        assert isinstance(per_day, pd.Series)
        assert np.isfinite(total)
        assert (per_day >= 0).all(), "wake after sleep onset cannot be negative"

    def test_broken_sleep_has_more_waso_than_uninterrupted_sleep(
        self, night_signal, broken_night_signal
    ):
        _, clean = waso(night_signal, frequency=ONE_MINUTE)
        _, broken = waso(broken_night_signal, frequency=ONE_MINUTE)
        assert broken > clean


# Sleep diary


@pytest.mark.needs_data
class TestSleepDiary:
    @pytest.fixture
    def diary(self, sleep_diary_ods, raw_awd_fresh):
        return SleepDiary(
            input_fname=str(sleep_diary_ods),
            start_time=raw_awd_fresh.start_time,
            periods=raw_awd_fresh.length(),
            frequency=raw_awd_fresh.frequency,
        )

    def test_loads_and_exposes_its_name(self, diary):
        assert isinstance(diary.name, str) and diary.name

    def test_diary_is_a_table_of_periods(self, diary):
        assert isinstance(diary.diary, pd.DataFrame)
        assert len(diary.diary) > 0

    def test_summary_is_a_table(self, diary):
        assert isinstance(diary.summary(), (pd.Series, pd.DataFrame))

    def test_default_state_vocabulary(self, diary):
        assert set(diary.state_index) == {"ACTIVE", "NAP", "NIGHT", "NOWEAR"}

    def test_total_times_return_a_mean_and_a_spread(self, diary):
        """Each ``total_*_time`` returns ``(mean, std)``, not a single duration."""
        for total in (
            diary.total_bed_time(),
            diary.total_nap_time(),
            diary.total_nowear_time(),
        ):
            assert isinstance(total, tuple) and len(total) == 2
            mean, spread = total
            assert mean >= pd.Timedelta(0)
            assert spread >= pd.Timedelta(0)

    def test_shapes_is_a_method_yielding_one_shape_per_period(self, diary):
        """``shapes`` reads like a property but must be called."""
        assert len(diary.shapes()) == len(diary.diary)

    def test_sleep_onset_latency_returns_per_night_values_and_a_mean(
        self, diary, raw_awd_fresh
    ):
        per_night, mean = diary.sleep_onset_latency(raw_awd_fresh.activity)
        assert isinstance(per_night, pd.Series)
        assert isinstance(mean, pd.Timedelta)
        assert len(per_night) > 0

    @pytest.mark.xfail(reason="SleepDiary.sleep_efficiency returns None", strict=True)
    def test_sleep_efficiency_is_a_fraction(self, diary, raw_awd_fresh):
        A.assert_within_range(
            diary.sleep_efficiency(raw_awd_fresh.activity), 0.0, 1.0, name="efficiency"
        )

    def test_sleep_efficiency_returning_none_is_pinned(self, diary, raw_awd_fresh):
        assert diary.sleep_efficiency(raw_awd_fresh.activity) is None

    @pytest.mark.xfail(
        reason="The bundled extra-states diary fails to load with the default state_index",
        raises=KeyError,
        strict=True,
    )
    def test_extra_states_file_loads(self, sleep_diary_extra_states_ods, raw_awd_fresh):
        SleepDiary(
            input_fname=str(sleep_diary_extra_states_ods),
            start_time=raw_awd_fresh.start_time,
            periods=raw_awd_fresh.length(),
            frequency=raw_awd_fresh.frequency,
        )

    def test_missing_file_raises(self, raw_awd_fresh, tmp_path):
        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            SleepDiary(
                input_fname=str(tmp_path / "absent.ods"),
                start_time=raw_awd_fresh.start_time,
                periods=raw_awd_fresh.length(),
                frequency=raw_awd_fresh.frequency,
            )
