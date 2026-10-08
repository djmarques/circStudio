"""Circular statistics in ``circstudio.analysis.models.tools``."""

import numpy as np
import pandas as pd
import pytest

from circstudio.analysis.models.tools import (
    Cir_Descriptive_Stats as CDS,
)
from circstudio.analysis.models.tools import (
    Cir_Inference_Stats as CIS,
)
from circstudio.analysis.models.tools import Tools

TWO_PI = 2 * np.pi


def hours(h: float) -> float:
    """Clock hours -> radians."""
    return h * TWO_PI / 24.0


class TestDifferenceHhmm:
    @pytest.mark.parametrize(
        "start,end,expected_h,expected_m,expected_dir",
        [
            ("08:00", "10:00", 2, 0, "+"),
            ("10:00", "08:00", 2, 0, "-"),
            # diff == 0 < wrap_around == 24 h, so the direction reads '+'.
            ("00:00", "00:00", 0, 0, "+"),
            ("09:00", "09:30", 0, 30, "+"),
        ],
    )
    def test_same_day_differences(self, start, end, expected_h, expected_m, expected_dir):
        h, m, direction = Tools.difference_hhmm(start, end)
        assert (h, m) == (expected_h, expected_m)
        assert direction == expected_dir

    def test_midnight_wrap_is_the_short_way_round(self):
        """23:30 -> 00:30 is one hour forward, not 23 hours backward."""
        h, m, direction = Tools.difference_hhmm("23:30", "00:30")
        assert (h, m) == (1, 0)
        assert direction == "+"

    def test_midnight_wrap_backwards(self):
        h, m, direction = Tools.difference_hhmm("00:30", "23:30")
        assert (h, m) == (1, 0)
        assert direction == "-"

    def test_antipodal_times_are_twelve_hours_apart(self):
        h, m, _ = Tools.difference_hhmm("00:00", "12:00")
        assert (h, m) == (12, 0)

    def test_difference_never_exceeds_twelve_hours(self):
        """By construction the smallest arc is taken, so |diff| <= 12 h."""
        for start_h in range(24):
            for end_h in range(24):
                h, m, _ = Tools.difference_hhmm(f"{start_h:02d}:00", f"{end_h:02d}:00")
                assert h + m / 60 <= 12.0

    def test_symmetry_of_magnitude(self):
        """Swapping the arguments preserves the magnitude and flips the sign."""
        h1, m1, d1 = Tools.difference_hhmm("22:55", "06:48")
        h2, m2, d2 = Tools.difference_hhmm("06:48", "22:55")
        assert (h1, m1) == (h2, m2)
        assert {d1, d2} == {"+", "-"}

    def test_malformed_input_raises(self):
        with pytest.raises(ValueError):
            Tools.difference_hhmm("not a time", "10:00")


class TestDifferenceDecimal:
    @pytest.mark.parametrize(
        "start,end,expected,expected_dir",
        [
            (8.0, 10.0, 2.0, "+"),
            (10.0, 8.0, 2.0, "-"),
            (23.5, 0.5, 1.0, "+"),
            (0.5, 23.5, 1.0, "-"),
            (0.0, 12.0, 12.0, "-"),
        ],
    )
    def test_decimal_differences(self, start, end, expected, expected_dir):
        diff, direction = Tools.difference_decimal(start, end)
        assert diff == pytest.approx(expected)
        assert direction == expected_dir

    def test_agrees_with_the_hhmm_variant(self):
        """The two APIs must describe the same geometry."""
        h, m, direction = Tools.difference_hhmm("22:55", "06:48")
        diff, direction_dec = Tools.difference_decimal(22 + 55 / 60, 6 + 48 / 60)
        assert diff == pytest.approx(h + m / 60, abs=1 / 60)
        assert direction == direction_dec


class TestCircularDistance:
    def test_identical_angles_are_zero_apart(self):
        assert Tools._circular_distance(1.0, 1.0) == pytest.approx(0.0)

    def test_wrap_around_takes_the_short_arc(self):
        """0.1 rad and (2pi - 0.1) rad are 0.2 rad apart, not 2pi - 0.2."""
        assert Tools._circular_distance(0.1, TWO_PI - 0.1) == pytest.approx(0.2)

    def test_maximum_distance_is_pi(self):
        assert Tools._circular_distance(0.0, np.pi) == pytest.approx(np.pi)
        rng = np.random.default_rng(0)
        a, b = rng.uniform(0, TWO_PI, 200), rng.uniform(0, TWO_PI, 200)
        assert (Tools._circular_distance(a, b) <= np.pi + 1e-12).all()

    def test_symmetric(self):
        assert Tools._circular_distance(0.3, 2.0) == pytest.approx(
            Tools._circular_distance(2.0, 0.3)
        )


class TestHhmmToH:
    @pytest.mark.parametrize(
        "text,expected", [("00:00", 0.0), ("07:09", 7.15), ("7:09", 7.15), ("23:59", 23 + 59 / 60)]
    )
    def test_conversion(self, text, expected):
        assert Tools.hhmm_to_h(text) == pytest.approx(expected)

    @pytest.mark.parametrize("bad", ["7:9", "0709", "7h09", "", "25:00:00", "abc"])
    def test_malformed_input_raises(self, bad):
        with pytest.raises(ValueError, match="HH:MM"):
            Tools.hhmm_to_h(bad)


class TestHoursRads:
    def test_twenty_four_hours_is_two_pi(self):
        assert Tools.hours_rads("hours -> rads", 24.0) == pytest.approx(TWO_PI)

    def test_twelve_hours_is_pi(self):
        assert Tools.hours_rads("hours -> rads", 12.0) == pytest.approx(np.pi)

    def test_round_trip_in_both_directions(self):
        for value in (0.0, 3.0, 6.5, 18.0, 23.99):
            rads = Tools.hours_rads("hours -> rads", value)
            assert Tools.hours_rads("rads -> hours", rads) == pytest.approx(value)

    def test_invalid_flag_raises(self):
        with pytest.raises(ValueError, match="Invalid flag"):
            Tools.hours_rads("hours->rads", 3.0)  # missing spaces


class TestResultantLength:
    def test_identical_angles_give_one(self):
        assert CDS.resultant_length(np.full(10, 1.234)) == pytest.approx(1.0)

    def test_uniformly_spread_angles_give_zero(self):
        angles = np.linspace(0, TWO_PI, 360, endpoint=False)
        assert CDS.resultant_length(angles) == pytest.approx(0.0, abs=1e-12)

    def test_antipodal_pair_gives_zero(self):
        assert CDS.resultant_length(np.array([0.0, np.pi])) == pytest.approx(0.0, abs=1e-12)

    def test_always_within_unit_interval(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            angles = rng.uniform(0, TWO_PI, rng.integers(2, 50))
            r = CDS.resultant_length(angles)
            assert 0.0 <= r <= 1.0 + 1e-12

    def test_rotation_invariant(self):
        """Concentration does not depend on where the data sit on the circle."""
        angles = np.array([0.1, 0.2, 0.3])
        r1 = CDS.resultant_length(angles)
        r2 = CDS.resultant_length(angles + 2.0)
        assert r1 == pytest.approx(r2)


class TestCircularVarianceAndStd:
    def test_variance_is_one_minus_resultant_length(self):
        angles = np.array([0.1, 0.5, 1.2, 3.0])
        assert CDS.circular_variance(angles) == pytest.approx(
            1 - CDS.resultant_length(angles)
        )

    def test_identical_angles_have_zero_variance(self):
        assert CDS.circular_variance(np.full(8, 2.0)) == pytest.approx(0.0)

    def test_uniform_angles_have_variance_one(self):
        angles = np.linspace(0, TWO_PI, 360, endpoint=False)
        assert CDS.circular_variance(angles) == pytest.approx(1.0, abs=1e-12)

    def test_variance_within_unit_interval(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            angles = rng.uniform(0, TWO_PI, 30)
            assert 0.0 <= CDS.circular_variance(angles) <= 1.0 + 1e-12

    def test_std_is_zero_for_identical_angles(self):
        """r == 1 -> sqrt(-2 ln 1) == 0."""
        assert CDS.circular_std(np.full(8, 2.0)) == pytest.approx(0.0, abs=1e-7)

    def test_std_grows_with_dispersion(self):
        tight = np.array([1.0, 1.01, 0.99, 1.005])
        loose = np.array([0.0, 1.5, 3.0, 4.5])
        assert CDS.circular_std(tight) < CDS.circular_std(loose)

    def test_zero_resultant_guard_is_effectively_dead_code(self):
        """``circular_std`` guards ``r > 0`` -- but r is never *exactly* 0."""
        antipodal = np.array([0.0, np.pi])
        r = CDS.resultant_length(antipodal)
        assert r != 0.0, "guard assumes exact zero, which floating point does not deliver"
        assert r == pytest.approx(0.0, abs=1e-15)

        std = CDS.circular_std(antipodal)
        assert np.isfinite(std)
        assert std > 5.0, "rounding noise, not a meaningful dispersion estimate"

    def test_zero_resultant_result_is_governed_by_machine_epsilon(self):
        """The returned std tracks log(machine epsilon), not the data."""
        antipodal = np.array([0.0, np.pi])
        uniform = np.linspace(0, TWO_PI, 360, endpoint=False)
        noise_driven_scale = np.sqrt(-2 * np.log(np.finfo(float).eps))
        for angles in (antipodal, uniform):
            assert CDS.circular_std(angles) == pytest.approx(noise_driven_scale, rel=0.6)


class TestCircularMean:
    def test_mean_of_identical_angles_is_that_angle(self):
        assert CDS.circular_mean(np.full(5, 1.0)) == pytest.approx(1.0)

    def test_mean_across_midnight_is_midnight(self):
        """23:00 and 01:00 average to midnight, not to noon."""
        angles = np.array([hours(23.0), hours(1.0)])
        mean_hours = Tools.hours_rads("rads -> hours", CDS.circular_mean(angles)) % 24
        assert mean_hours == pytest.approx(0.0, abs=1e-9) or mean_hours == pytest.approx(
            24.0, abs=1e-9
        )

    def test_mean_is_not_the_arithmetic_mean_across_midnight(self):
        angles = np.array([hours(23.0), hours(1.0)])
        mean_hours = Tools.hours_rads("rads -> hours", CDS.circular_mean(angles)) % 24
        assert abs(mean_hours - 12.0) > 11.0, "must not return the naive arithmetic mean"

    def test_output_is_wrapped_into_zero_to_two_pi(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            angles = rng.uniform(-10, 10, 25)
            mean = CDS.circular_mean(angles)
            assert 0.0 <= mean < TWO_PI

    def test_rotation_equivariance(self):
        """Rotating every angle by delta rotates the mean by delta."""
        angles = np.array([0.2, 0.5, 0.9])
        delta = 1.3
        rotated = CDS.circular_mean(angles + delta)
        expected = (CDS.circular_mean(angles) + delta) % TWO_PI
        assert rotated == pytest.approx(expected)

    def test_mean_of_symmetric_set_is_the_axis_of_symmetry(self):
        centre = 2.0
        angles = np.array([centre - 0.4, centre, centre + 0.4])
        assert CDS.circular_mean(angles) == pytest.approx(centre)


class TestCircularMedian:
    def test_even_length_input_works(self):
        angles = np.array([1.0, 1.1, 1.2, 1.3])
        median = CDS.circular_median(angles)
        assert 1.0 <= median <= 1.3

    def test_even_length_is_the_mean_of_the_two_most_central(self):
        angles = np.array([1.0, 1.1, 1.2, 1.3])
        assert CDS.circular_median(angles) == pytest.approx((1.1 + 1.2) / 2)

    @pytest.mark.xfail(
        reason="circular_median raises NameError on odd-length input (undefined name 'signal')",
        raises=NameError,
        strict=True,
    )
    def test_odd_length_input_works(self):
        angles = np.array([1.0, 1.1, 1.2])
        assert CDS.circular_median(angles) == pytest.approx(1.1)

    def test_even_length_requires_an_ndarray_not_a_list(self):
        """`angle_vector[sorted_dist[0:2]]` is fancy indexing, so a plain list fails."""
        with pytest.raises(TypeError):
            CDS.circular_median([1.0, 1.1, 1.2, 1.3])


class TestPermutationTest:
    @staticmethod
    def _frame(seed=0, separated=True):
        rng = np.random.default_rng(seed)
        if separated:
            # One tightly concentrated group, one uniformly spread group.
            a = rng.normal(1.0, 0.05, 40) % TWO_PI
            b = rng.uniform(0, TWO_PI, 40)
        else:
            a = rng.uniform(0, TWO_PI, 40)
            b = rng.uniform(0, TWO_PI, 40)
        return pd.DataFrame({"group_a": a, "group_b": b})

    @pytest.mark.xfail(
        reason="permutation_test raises NameError: 'cds' is undefined (should be 'cls')",
        raises=NameError,
        strict=True,
    )
    def test_permutation_test_runs_at_all(self):
        p_values = CIS.permutation_test(self._frame(), permutations=50)
        assert len(p_values) == 2

    @pytest.mark.xfail(
        reason="Blocked by the permutation_test NameError",
        raises=NameError,
        strict=True,
    )
    def test_p_values_lie_in_the_unit_interval(self):
        p_values = CIS.permutation_test(self._frame(), permutations=100)
        assert all(0.0 <= p <= 1.0 for p in p_values)

    @pytest.mark.xfail(
        reason="Blocked by the permutation_test NameError",
        raises=NameError,
        strict=True,
    )
    def test_concentrated_group_separates_from_uniform_group(self):
        p_values = CIS.permutation_test(self._frame(separated=True), permutations=200)
        # The tightly concentrated group has much lower variance than chance.
        assert p_values[0] > p_values[1]

    @pytest.mark.xfail(
        reason="Blocked by the permutation_test NameError; the function is also unseeded",
        raises=NameError,
        strict=True,
    )
    def test_results_are_reproducible(self):
        frame = self._frame()
        assert CIS.permutation_test(frame, permutations=50) == CIS.permutation_test(
            frame, permutations=50
        )


class TestInheritanceContract:
    def test_inference_stats_inherits_descriptive_methods(self):
        angles = np.array([0.1, 0.2, 0.3])
        assert CIS.circular_mean(angles) == CDS.circular_mean(angles)
        assert CIS.resultant_length(angles) == CDS.resultant_length(angles)

    def test_descriptive_stats_inherits_time_tools(self):
        assert CDS.hhmm_to_h("06:30") == pytest.approx(6.5)
