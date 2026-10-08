"""Light schedules and the circadian oscillator models."""

import numpy as np
import pytest

from circstudio.analysis import (
    ESRI,
    Breslow13,
    Forger,
    HannaySP,
    HannayTP,
    Hilaire07,
    Jewett,
    Light,
    Model,
    ModelComparer,
    Skeldon23,
)
from helpers import assertions as A

# Models sharing the (data, inputs, time) constructor and the integrate() API.
OSCILLATORS = [Forger, Jewett, HannaySP, HannayTP]

BINS_PER_HOUR = 6  # 10-minute resolution keeps the ODE integrations quick


@pytest.fixture(scope="module")
def light_schedule():
    """6 days of 16 h light / 8 h dark at 10-minute resolution."""
    return Light.create(
        total_days=6,
        light_on_hours=16,
        bins_per_hour=BINS_PER_HOUR,
        low=0,
        high=1000,
    )


@pytest.fixture(scope="module")
def darkness():
    """Constant darkness -- the free-running condition."""
    return Light.create(
        total_days=6, light_on_hours=16, bins_per_hour=BINS_PER_HOUR, low=0, high=0
    )


# Light


class TestLightCreate:
    def test_sample_count_follows_days_and_resolution(self):
        light = Light.create(total_days=6, bins_per_hour=BINS_PER_HOUR)
        assert len(light.light_vector) == 6 * 24 * BINS_PER_HOUR

    def test_time_vector_spans_the_requested_days_in_hours(self):
        light = Light.create(total_days=2, bins_per_hour=BINS_PER_HOUR)
        time = np.asarray(light.time_vector)
        assert time[0] == pytest.approx(0.0)
        assert time[-1] == pytest.approx(48.0 - 1 / BINS_PER_HOUR)

    def test_dark_phase_is_exactly_the_low_level(self):
        """The dark phase is the part of the schedule that is exact."""
        light = Light.create(
            total_days=2, light_on_hours=16, bins_per_hour=BINS_PER_HOUR, low=0, high=1000
        )
        values = np.asarray(light.light_vector)
        n_dark = int((values == 0).sum())
        assert n_dark == 2 * (24 - 16) * BINS_PER_HOUR

    def test_longer_photoperiod_means_less_darkness(self):
        short = Light.create(total_days=2, light_on_hours=8, bins_per_hour=BINS_PER_HOUR)
        long = Light.create(total_days=2, light_on_hours=20, bins_per_hour=BINS_PER_HOUR)
        assert (np.asarray(short.light_vector) == 0).sum() > (
            np.asarray(long.light_vector) == 0
        ).sum()

    def test_light_never_exceeds_the_requested_maximum(self):
        light = Light.create(total_days=2, bins_per_hour=BINS_PER_HOUR, low=0, high=1000)
        values = np.asarray(light.light_vector)
        assert values.min() >= 0
        assert values.max() <= 1000

    @pytest.mark.xfail(
        reason="Light.create fills the light phase with random values instead of the constant high level",
        strict=True,
    )
    def test_light_phase_is_the_constant_high_level(self):
        light = Light.create(
            total_days=2, light_on_hours=16, bins_per_hour=BINS_PER_HOUR, low=0, high=1000
        )
        values = np.asarray(light.light_vector)
        assert int((values == 1000).sum()) == 2 * 16 * BINS_PER_HOUR

    def test_the_random_light_phase_is_pinned(self):
        """Pin current behaviour so the xfail above is unambiguous."""
        light = Light.create(
            total_days=2, light_on_hours=16, bins_per_hour=BINS_PER_HOUR, low=0, high=1000
        )
        values = np.asarray(light.light_vector)
        lit = values[values > 0]
        # One random level per lit bin (16 h x 6 bins), tiled across days
        assert len(np.unique(lit)) == 16 * BINS_PER_HOUR

    def test_two_calls_give_different_light(self):
        """A direct consequence of the randomised light phase: irreproducibility."""
        a = np.asarray(Light.create(total_days=1, bins_per_hour=BINS_PER_HOUR).light_vector)
        b = np.asarray(Light.create(total_days=1, bins_per_hour=BINS_PER_HOUR).light_vector)
        assert not np.array_equal(a, b)


class TestLightOperations:
    def test_with_datetime_index_produces_a_regular_series(self):
        light = Light.create(total_days=2, bins_per_hour=BINS_PER_HOUR)
        series = light.with_datetime_index(start="2020-01-01")
        A.assert_datetime_index_regular(series, name="light")
        assert len(series) == 2 * 24 * BINS_PER_HOUR

    def test_scalar_multiplication_scales_every_sample(self):
        light = Light.create(total_days=1, bins_per_hour=BINS_PER_HOUR, low=0, high=100)
        original = np.asarray(light.light_vector).copy()
        scaled = np.asarray((light * 3).light_vector)
        np.testing.assert_allclose(scaled, original * 3)

    def test_division_inverts_multiplication(self):
        light = Light.create(total_days=1, bins_per_hour=BINS_PER_HOUR, low=0, high=100)
        original = np.asarray(light.light_vector).copy()
        round_tripped = np.asarray(((light * 4) / 4).light_vector)
        np.testing.assert_allclose(round_tripped, original)

    def test_str_is_informative(self):
        assert str(Light.create(total_days=1, bins_per_hour=BINS_PER_HOUR))

    @pytest.mark.xfail(
        reason="Light.downsample raises TypeError because the slice step is not cast to int",
        raises=TypeError,
        strict=True,
    )
    def test_downsample_reduces_the_sample_count(self):
        light = Light.create(total_days=2, bins_per_hour=BINS_PER_HOUR)
        before = len(light.light_vector)
        light.downsample(2)
        assert len(light.light_vector) == before // 2


# Published parameter defaults


class TestPublishedDefaults:
    """A silent typo in a published constant changes every result invisibly."""

    @staticmethod
    def defaults(cls):
        import inspect

        return {
            name: param.default
            for name, param in inspect.signature(cls.__init__).parameters.items()
            if param.default is not inspect.Parameter.empty
        }

    def test_forger_matches_forger_1999(self):
        d = self.defaults(Forger)
        assert d["taux"] == 24.2
        assert d["mu"] == 0.23
        assert d["g"] == 33.75
        assert d["alpha_0"] == 0.05
        assert d["beta"] == 0.0075
        assert d["p"] == 0.50
        assert d["i0"] == 9500.0
        assert d["k"] == 0.55
        assert d["cbt_to_dlmo"] == 7.0

    def test_jewett_matches_jewett_1999(self):
        d = self.defaults(Jewett)
        assert d["taux"] == 24.2
        assert d["mu"] == 0.13
        assert d["g"] == 19.875
        assert d["beta"] == 0.013
        assert d["k"] == 0.55
        assert d["q"] == pytest.approx(1.0 / 3.0)
        assert d["i0"] == 9500
        assert d["p"] == 0.6
        assert d["alpha_0"] == 0.16
        assert d["phi_ref"] == 0.8

    def test_hannay_single_population_matches_hannay_2019(self):
        d = self.defaults(HannaySP)
        assert d["tau"] == 23.84
        assert d["k"] == 0.06358
        assert d["gamma"] == 0.024
        assert d["beta"] == -0.09318
        assert d["a1"] == 0.3855
        assert d["a2"] == 0.1977

    def test_hannay_two_population_matches_hannay_2019(self):
        d = self.defaults(HannayTP)
        assert d["tauv"] == 24.25
        assert d["taud"] == 24.0
        assert d["kvv"] == 0.05
        assert d["kdd"] == 0.04
        assert d["kvd"] == 0.05
        assert d["kdv"] == 0.01
        assert d["gamma"] == 0.024

    def test_hilaire_matches_st_hilaire_2007(self):
        d = self.defaults(Hilaire07)
        assert d["taux"] == 24.2
        assert d["g"] == 37.0
        assert d["k"] == 0.55
        assert d["mu"] == 0.13
        assert d["beta"] == 0.007
        assert d["rho"] == 0.032
        assert d["i0"] == 9500.0

    def test_breslow_matches_breslow_2013(self):
        d = self.defaults(Breslow13)
        assert d["k"] == 0.55
        assert d["i0"] == 9500.0
        assert d["i1"] == 100.0
        assert d["alpha_0"] == 0.1
        assert d["beta"] == 0.007
        assert d["g"] == 37.0

    def test_skeldon_matches_skeldon_2023(self):
        d = self.defaults(Skeldon23)
        assert d["mu"] == 17.78
        assert d["chi"] == 45.0
        assert d["h0"] == 13.0
        assert d["tauc"] == 24.2
        assert d["g"] == 19.9

    @pytest.mark.parametrize(
        "cls,tau_name,tau_value",
        [(Forger, "taux", 24.2), (Jewett, "taux", 24.2), (HannaySP, "tau", 23.84)],
        ids=["Forger", "Jewett", "HannaySP"],
    )
    def test_intrinsic_period_is_near_twenty_four_hours(self, cls, tau_name, tau_value):
        """Sanity floor: a human circadian period must be close to a day."""
        assert 23.0 < self.defaults(cls)[tau_name] < 25.5
        assert self.defaults(cls)[tau_name] == tau_value


# Integration behaviour


class TestOscillatorIntegration:
    @pytest.mark.parametrize("cls", OSCILLATORS, ids=lambda c: c.__name__)
    def test_integrates_without_producing_non_finite_states(self, cls, light_schedule):
        model = cls(inputs=light_schedule.light_vector, time=light_schedule.time_vector)
        model.integrate()
        assert np.isfinite(np.asarray(model.amplitude())).all()

    @pytest.mark.parametrize("cls", OSCILLATORS, ids=lambda c: c.__name__)
    def test_amplitude_is_non_negative(self, cls, light_schedule):
        model = cls(inputs=light_schedule.light_vector, time=light_schedule.time_vector)
        model.integrate()
        assert (np.asarray(model.amplitude()) >= 0).all()

    @pytest.mark.parametrize("cls", OSCILLATORS, ids=lambda c: c.__name__)
    def test_amplitude_is_physiologically_bounded(self, cls, light_schedule):
        """These models are scaled so amplitude sits near unity, never exploding."""
        model = cls(inputs=light_schedule.light_vector, time=light_schedule.time_vector)
        model.integrate()
        assert np.asarray(model.amplitude()).max() < 10.0

    @pytest.mark.parametrize("cls", OSCILLATORS, ids=lambda c: c.__name__)
    def test_phase_and_cbt_are_finite_arrays(self, cls, light_schedule):
        model = cls(inputs=light_schedule.light_vector, time=light_schedule.time_vector)
        model.integrate()
        assert np.isfinite(np.asarray(model.phase())).all()
        assert np.asarray(model.cbt()).size > 0

    @pytest.mark.parametrize("cls", OSCILLATORS, ids=lambda c: c.__name__)
    def test_integration_is_deterministic(self, cls, light_schedule):
        """Same light in, same trajectory out -- the models carry no randomness."""
        results = []
        for _ in range(2):
            model = cls(inputs=light_schedule.light_vector, time=light_schedule.time_vector)
            model.integrate()
            results.append(np.asarray(model.amplitude()))
        np.testing.assert_allclose(results[0], results[1])

    @pytest.mark.parametrize("cls", OSCILLATORS, ids=lambda c: c.__name__)
    def test_darkness_and_light_give_different_trajectories(
        self, cls, light_schedule, darkness
    ):
        """If the light input did not matter, the model would be inert."""
        lit = cls(inputs=light_schedule.light_vector, time=light_schedule.time_vector)
        lit.integrate()
        dark = cls(inputs=darkness.light_vector, time=darkness.time_vector)
        dark.integrate()
        assert not np.allclose(
            np.asarray(lit.amplitude()), np.asarray(dark.amplitude())
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("cls", OSCILLATORS, ids=lambda c: c.__name__)
    def test_amplitude_settles_under_a_stable_light_dark_cycle(self, cls):
        """Entrainment: the oscillator must stop drifting, not run away."""
        schedule = Light.create(
            total_days=12, light_on_hours=16, bins_per_hour=BINS_PER_HOUR, low=0, high=1000
        )
        model = cls(inputs=schedule.light_vector, time=schedule.time_vector)
        model.integrate()
        amplitude = np.ravel(np.asarray(model.amplitude()))
        per_day = 24 * BINS_PER_HOUR
        first_day = np.ptp(amplitude[:per_day])
        last_day = np.ptp(amplitude[-per_day:])
        assert last_day <= first_day * 3 + 1e-9, "amplitude excursion must not blow up"

    def test_base_model_exposes_the_documented_interface(self):
        for name in (
            "initialize_model_states",
            "integrate",
            "get_initial_conditions",
            "dlmos",
            "plot",
        ):
            assert hasattr(Model, name), f"Model.{name} is missing"


class TestModelValidation:
    @pytest.mark.xfail(
        reason="Forger integrates light and time vectors of different lengths without complaint",
        strict=True,
    )
    def test_mismatched_input_and_time_lengths_are_rejected(self, light_schedule):
        short_time = np.asarray(light_schedule.time_vector)[:10]
        with pytest.raises(Exception):
            model = Forger(inputs=light_schedule.light_vector, time=short_time)
            model.integrate()

    @pytest.mark.xfail(reason="An empty light input is integrated instead of rejected", strict=True)
    def test_empty_light_input_is_rejected(self):
        with pytest.raises(Exception):
            model = Forger(inputs=np.array([]), time=np.array([]))
            model.integrate()


# ESRI and ModelComparer


class TestESRI:
    def test_produces_a_finite_series_for_a_regular_schedule(self, light_schedule):
        esri = ESRI(
            inputs=light_schedule.light_vector,
            time=light_schedule.time_vector,
            window_size_days=2.0,
            esri_time_step_hours=6.0,
        )
        result = esri.calculate()
        assert result is not None
        assert np.isfinite(np.asarray(result, dtype=float)).any()

    def test_step_size_controls_the_output_resolution(self, light_schedule):
        coarse = ESRI(
            inputs=light_schedule.light_vector,
            time=light_schedule.time_vector,
            window_size_days=2.0,
            esri_time_step_hours=12.0,
        ).calculate()
        fine = ESRI(
            inputs=light_schedule.light_vector,
            time=light_schedule.time_vector,
            window_size_days=2.0,
            esri_time_step_hours=3.0,
        ).calculate()
        assert len(np.ravel(np.asarray(fine))) > len(np.ravel(np.asarray(coarse)))

    def test_default_parameters_are_the_documented_ones(self):
        import inspect

        params = inspect.signature(ESRI.__init__).parameters
        assert params["window_size_days"].default == 4.0
        assert params["esri_time_step_hours"].default == 1.0
        assert params["initial_amplitude"].default == 0.1


class TestModelComparer:
    @pytest.fixture
    def comparer(self, light_schedule):
        return ModelComparer(
            inputs=light_schedule.light_vector,
            time=light_schedule.time_vector,
            equilibrate=False,
            loop_number=1,
        )

    def test_rmse_is_non_negative(self, comparer):
        comparer.predict_forger()
        value = comparer.rmse()
        assert np.all(np.asarray(value, dtype=float) >= 0)

    def test_cumulative_rmse_is_a_running_error_not_a_running_sum(self, comparer):
        """``cumulative_rmse`` is an RMSE over a growing window, so it may fall."""
        comparer.predict_forger()
        cumulative = np.ravel(np.asarray(comparer.cumulative_rmse(), dtype=float))
        assert (cumulative >= 0).all()
        assert np.isfinite(cumulative).all()
        assert (np.diff(cumulative) < 0).any(), "it genuinely decreases somewhere"

    def test_half_life_is_finite(self, comparer):
        comparer.predict_forger()
        comparer.cumulative_rmse()
        assert np.isfinite(float(np.ravel(np.asarray(comparer.half_life_crmse()))[0]))

    def test_default_multipliers_are_neutral(self):
        import inspect

        params = inspect.signature(ModelComparer.__init__).parameters
        assert params["a1"].default == 1.0
        assert params["a2"].default == 1.0
        assert params["m1"].default == 0.0
        assert params["m2"].default == 0.0
