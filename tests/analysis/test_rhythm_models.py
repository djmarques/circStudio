"""Rhythm models: Cosinor, SSA, FLM, LIDS and fractal analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from circstudio.analysis import FLM, LIDS, SSA, Cosinor, Fractal
from circstudio.analysis.lids.transforms import _lids_func, _lids_inverse_func
from helpers import assertions as A
from helpers import signals as S


# Cosinor


class TestCosinorDefaults:
    def test_default_initial_parameters(self):
        """Carried over from the legacy tests/test_cosinor.py."""
        params = Cosinor().fit_initial_params
        assert params["Amplitude"].value == 50
        assert params["Acrophase"].value == pytest.approx(np.pi)
        assert params["Period"].value == 1440
        assert params["Mesor"].value == 50

    def test_parameter_bounds(self):
        params = Cosinor().fit_initial_params
        assert params["Amplitude"].min == 0
        assert params["Acrophase"].min == 0
        assert params["Acrophase"].max == pytest.approx(2 * np.pi)
        assert params["Period"].min == 0

    def test_initial_parameters_can_be_replaced(self):
        from lmfit import Parameters

        model = Cosinor()
        custom = Parameters()
        custom.add("Acrophase", value=1.0, min=-np.pi, max=np.pi)
        custom.add("Amplitude", value=25, min=0)
        custom.add("Period", value=720, min=0)
        custom.add("Mesor", value=10, min=0)
        model.fit_initial_params = custom
        assert model.fit_initial_params["Period"].value == 720
        assert model.fit_initial_params["Mesor"].value == 10


class TestCosinorRecovery:
    """A noiseless cosine must give its own parameters back."""

    @pytest.mark.parametrize("acrophase_hours", [2, 6, 12, 18, 22])
    def test_recovers_amplitude_mesor_and_period(self, acrophase_hours):
        series = S.sinewave(
            n_days=7, amplitude=100.0, mesor=100.0, acrophase_hours=acrophase_hours
        )
        result = Cosinor().fit(series)
        assert result.params["Amplitude"].value == pytest.approx(100.0, rel=1e-3)
        assert result.params["Mesor"].value == pytest.approx(100.0, rel=1e-3)
        assert result.params["Period"].value == pytest.approx(1440.0, rel=1e-3)

    @pytest.mark.parametrize("acrophase_hours", [2, 6, 12, 18])
    def test_recovers_the_acrophase(self, acrophase_hours):
        r"""A peak at hour h corresponds to ``phi = -2*pi*h/24`` wrapped into [0, 2*pi)."""
        series = S.sinewave(
            n_days=7, amplitude=100.0, mesor=100.0, acrophase_hours=acrophase_hours
        )
        result = Cosinor().fit(series)
        expected = (-2 * np.pi * acrophase_hours / 24) % (2 * np.pi)
        assert result.params["Acrophase"].value == pytest.approx(expected, abs=1e-3)

    def test_offset_shifts_only_the_mesor(self):
        base = S.sinewave(n_days=7, amplitude=100.0, mesor=100.0, acrophase_hours=6)
        shifted = Cosinor().fit(base + 50.0)
        assert shifted.params["Mesor"].value == pytest.approx(150.0, rel=1e-3)
        assert shifted.params["Amplitude"].value == pytest.approx(100.0, rel=1e-3)

    def test_scaling_scales_mesor_and_amplitude_together(self):
        base = S.sinewave(n_days=7, amplitude=100.0, mesor=100.0, acrophase_hours=6)
        scaled = Cosinor().fit(base * 3.0)
        assert scaled.params["Amplitude"].value == pytest.approx(300.0, rel=1e-3)
        assert scaled.params["Mesor"].value == pytest.approx(300.0, rel=1e-3)

    def test_recovery_under_noise_within_the_snr_bound(self):
        """Tolerance derived from the SNR, not chosen by eye."""
        series = S.sinewave(
            n_days=7, amplitude=100.0, mesor=100.0, acrophase_hours=6, noise_sd=20.0
        )
        result = Cosinor().fit(series)
        n = len(series)
        five_sigma = 5 * 20.0 * np.sqrt(2 / n)
        assert result.params["Amplitude"].value == pytest.approx(100.0, abs=five_sigma)
        assert result.params["Mesor"].value == pytest.approx(100.0, abs=five_sigma)

    def test_flat_signal_has_no_amplitude(self):
        result = Cosinor().fit(S.flat(n_days=7, value=50.0))
        assert result.params["Amplitude"].value == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.xfail(
        reason="Cosinor cannot reach an acrophase on its lower bound (peak at recording start)",
        strict=True,
    )
    def test_acrophase_at_the_recording_start_is_recovered(self):
        series = S.sinewave(
            n_days=7, amplitude=100.0, mesor=100.0, acrophase_hours=0.0
        )
        result = Cosinor().fit(series)
        assert result.params["Amplitude"].value == pytest.approx(100.0, rel=1e-2)

    def test_the_boundary_failure_mode_is_pinned(self):
        """Document the exact wrong answer, so the xfail above is unambiguous."""
        series = S.sinewave(n_days=7, amplitude=100.0, mesor=100.0, acrophase_hours=0.0)
        result = Cosinor().fit(series)
        assert result.params["Amplitude"].value == pytest.approx(0.0, abs=1e-6)
        assert result.params["Mesor"].value == pytest.approx(100.0, rel=1e-2)


class TestCosinorOutputs:
    @pytest.fixture(scope="class")
    @classmethod
    def fitted(cls):
        series = S.sinewave(n_days=5, amplitude=80.0, mesor=120.0, acrophase_hours=8)
        model = Cosinor()
        return series, model, model.fit(series)

    def test_best_fit_aligns_with_the_input(self, fitted):
        series, model, result = fitted
        curve = model.best_fit(series, result.params)
        assert isinstance(curve, pd.Series)
        pd.testing.assert_index_equal(curve.index, series.index)

    def test_best_fit_reproduces_a_noiseless_signal(self, fitted):
        series, model, result = fitted
        curve = model.best_fit(series, result.params)
        np.testing.assert_allclose(curve.values, series.values, atol=1e-3)

    def test_plot_returns_a_figure_with_data_and_fit(self, fitted):
        series, model, result = fitted
        fig = model.plot(series, result.params)
        assert isinstance(fig, go.Figure)
        A.assert_figure_renders(fig, min_traces=2)

    def test_shorter_than_one_period_still_returns_a_result(self, fitted):
        """Under-determined, but it must not crash silently mid-pipeline."""
        short = S.sinewave(n_days=7)[:600]  # 10 h < 24 h period
        result = Cosinor().fit(short)
        assert result is not None

    def test_nan_policy_raise_rejects_missing_data(self):
        series = S.signal_with_gap(n_days=3, gap_start_epoch=100, gap_length_epochs=20)
        with pytest.raises(ValueError):
            Cosinor().fit(series, nan_policy="raise")

    def test_nan_policy_omit_tolerates_missing_data(self):
        series = S.signal_with_gap(n_days=3, gap_start_epoch=100, gap_length_epochs=20)
        result = Cosinor().fit(series, nan_policy="omit")
        assert np.isfinite(result.params["Mesor"].value)


# SSA -- Singular Spectrum Analysis


@pytest.fixture(scope="module")
def ssa_signal():
    """24 h + 4 h tones at 10 min resolution: 432 samples, SVD is cheap."""
    return S.two_tone(n_days=3, sampling_period=600)


@pytest.fixture(scope="module")
def fitted_ssa(ssa_signal):
    ssa = SSA(ssa_signal, window_length="12h")
    ssa.fit()
    return ssa


class TestSSADecomposition:
    def test_trajectory_matrix_dimensions(self, fitted_ssa, ssa_signal):
        traj = fitted_ssa.build_trajectory_matrix()
        assert traj.shape == (fitted_ssa.L, fitted_ssa.K)
        assert fitted_ssa.L + fitted_ssa.K - 1 == len(ssa_signal)

    def test_trajectory_matrix_is_hankel(self, fitted_ssa):
        """Constant anti-diagonals -- the defining property of the embedding."""
        traj = fitted_ssa.build_trajectory_matrix()
        for offset in (0, 1, 5, -3):
            diagonal = np.diagonal(np.fliplr(traj), offset=offset)
            np.testing.assert_allclose(diagonal, diagonal[0], atol=1e-12)

    def test_variance_explained_is_a_normalised_descending_spectrum(self, fitted_ssa):
        variance = np.asarray(fitted_ssa.variance_explained)
        assert variance.sum() == pytest.approx(1.0, abs=1e-10)
        assert (variance >= -1e-15).all()
        A.assert_monotonic(variance, increasing=False)

    def test_singular_values_are_non_negative(self, fitted_ssa):
        assert (np.diag(fitted_ssa.sigma) >= 0).all()

    def test_variance_explained_matches_the_component_energy(self, fitted_ssa):
        """Each component's share must equal its share of the squared norm."""
        trajectory = fitted_ssa.build_trajectory_matrix()
        total_energy = np.sum(trajectory**2)
        for r in range(4):
            component = fitted_ssa.get_component_matrix(r)
            # Tolerance covers float error from rebuilding the components
            assert fitted_ssa.variance_explained[r] == pytest.approx(
                np.sum(component**2) / total_energy, rel=1e-6
            )

    def test_two_tone_variance_ratio_follows_the_squared_amplitudes(self, fitted_ssa):
        """Independent ground truth: energy scales with amplitude squared."""
        variance = np.asarray(fitted_ssa.variance_explained)
        slow_pair = variance[0] + variance[1]
        fast_pair = variance[2] + variance[3]
        assert slow_pair / fast_pair == pytest.approx((100 / 30) ** 2, rel=0.25)

    def test_full_reconstruction_returns_the_input(self, fitted_ssa, ssa_signal):
        """The single most valuable SSA test: the decomposition is complete."""
        rebuilt = fitted_ssa.reconstruct_signal(range(len(fitted_ssa.variance_explained)))
        assert isinstance(rebuilt, pd.Series)
        pd.testing.assert_index_equal(rebuilt.index, ssa_signal.index)
        np.testing.assert_allclose(rebuilt.values, ssa_signal.values, atol=1e-5)

    def test_partial_reconstruction_is_not_the_whole_signal(self, fitted_ssa, ssa_signal):
        leading = fitted_ssa.reconstruct_signal(range(2))
        assert np.abs(leading.values - ssa_signal.values).max() > 1e-3

    def test_leading_components_dominate_a_two_tone_signal(self, fitted_ssa):
        """Two pure tones concentrate almost all variance in four components."""
        variance = np.asarray(fitted_ssa.variance_explained)
        assert variance[:4].sum() > 0.99

    def test_component_matrix_has_trajectory_dimensions(self, fitted_ssa):
        assert fitted_ssa.get_component_matrix(0).shape == (fitted_ssa.L, fitted_ssa.K)

    def test_component_index_out_of_range_raises(self, fitted_ssa):
        with pytest.raises((ValueError, IndexError)):
            fitted_ssa.get_component_matrix(10_000)


class TestSSAWCorrelation:
    def test_matrix_is_square_symmetric_with_unit_diagonal(self, fitted_ssa):
        w = fitted_ssa.w_correlation_matrix(6)
        assert w.shape == (6, 6)
        np.testing.assert_allclose(np.diag(w), 1.0, atol=1e-9)
        np.testing.assert_allclose(w, w.T, atol=1e-9)

    def test_oscillatory_pairs_are_correlated_and_distinct_tones_are_not(
        self, fitted_ssa
    ):
        """A sinusoid occupies an eigentriple *pair*."""
        w = fitted_ssa.w_correlation_matrix(6)
        assert abs(w[0, 1]) > 0.5, "the paired component of a tone must correlate"
        assert abs(w[0, 2]) < 0.2, "components of different tones must not"


class TestSSAValidation:
    @pytest.mark.xfail(
        reason="SSA accepts a window longer than the series and returns an empty spectrum",
        strict=True,
    )
    def test_window_longer_than_the_series_is_rejected(self, ssa_signal):
        with pytest.raises((ValueError, IndexError)):
            SSA(ssa_signal, window_length="500h").fit()

    def test_the_degenerate_oversized_window_is_pinned(self, ssa_signal):
        ssa = SSA(ssa_signal, window_length="500h")
        ssa.fit()
        assert ssa.K < 0, "K = N - L + 1 goes negative once L exceeds N"
        assert len(ssa.variance_explained) == 0

    def test_series_without_a_frequency_is_rejected(self, ssa_signal):
        unset = ssa_signal.copy()
        unset.index.freq = None
        with pytest.raises((ValueError, AttributeError, TypeError)):
            SSA(unset, window_length="12h")


# FLM -- Functional Linear Modelling


@pytest.fixture(scope="module")
def flm_signal():
    return S.realistic_rest_activity(n_days=7, sampling_period=300)


class TestFLM:
    def test_fourier_basis_fits_and_evaluates(self, flm_signal):
        model = FLM(basis="fourier", sampling_freq="5min", max_order=9)
        model.fit(flm_signal)
        assert model.beta is not None
        evaluated = model.evaluate(r=10)
        assert isinstance(evaluated, np.ndarray)
        assert len(evaluated) > 0
        assert np.isfinite(evaluated).all()

    def test_spline_basis_fits(self, flm_signal):
        """Splines cap at degree 5 (scipy limit), unlike the Fourier basis."""
        model = FLM(basis="spline", sampling_freq="5min", max_order=5)
        model.fit(flm_signal)
        assert model.beta is not None

    def test_spline_degree_above_five_is_rejected(self, flm_signal):
        model = FLM(basis="spline", sampling_freq="5min", max_order=9)
        with pytest.raises(TypeError, match="degree of the spline"):
            model.fit(flm_signal)

    def test_unknown_basis_is_rejected(self):
        with pytest.raises(ValueError, match='"fourier" or "spline"'):
            FLM(basis="legendre", sampling_freq="5min", max_order=4)

    def test_higher_order_never_fits_worse(self, flm_signal):
        """Adding basis functions cannot increase least-squares residual."""
        errors = []
        for order in (3, 7, 11):
            model = FLM(basis="fourier", sampling_freq="5min", max_order=order)
            model.fit(flm_signal)
            smoothed = model.smooth_daily_profile(flm_signal)
            profile = flm_signal.groupby(
                [flm_signal.index.hour, flm_signal.index.minute]
            ).mean().values
            errors.append(float(np.sqrt(np.mean((smoothed - profile) ** 2))))
        A.assert_monotonic(errors, increasing=False)

    def test_smoothing_preserves_the_mean_and_reduces_variance(self, flm_signal):
        model = FLM(basis="fourier", sampling_freq="5min", max_order=9)
        model.fit(flm_signal)
        smoothed = model.smooth_timeseries(flm_signal)
        assert len(smoothed) == len(flm_signal)
        assert smoothed.mean() == pytest.approx(flm_signal.mean(), rel=0.05)
        assert smoothed.std() < flm_signal.std()

    def test_daily_profile_smoothing_has_one_day_of_samples(self, flm_signal):
        model = FLM(basis="fourier", sampling_freq="5min", max_order=9)
        model.fit(flm_signal)
        smoothed = model.smooth_daily_profile(flm_signal)
        assert len(smoothed) == 288  # 24 h at 5 min

    def test_fit_is_deterministic(self, flm_signal):
        """``beta`` is a dict keyed by coefficient name, not a bare array."""
        results = []
        for _ in range(2):
            model = FLM(basis="fourier", sampling_freq="5min", max_order=9)
            model.fit(flm_signal)
            assert isinstance(model.beta, dict)
            results.append({k: np.asarray(v) for k, v in model.beta.items()})
        assert results[0].keys() == results[1].keys()
        for key in results[0]:
            np.testing.assert_allclose(results[0][key], results[1][key])


# LIDS -- Locomotor Inactivity During Sleep


class TestLIDSTransforms:
    def test_forward_and_inverse_are_mutual_inverses(self):
        values = np.array([0.0, 1.0, 10.0, 100.0, 1000.0])
        np.testing.assert_allclose(
            _lids_inverse_func(_lids_func(values)), values, rtol=1e-9
        )

    def test_transform_maps_high_activity_to_low_lids(self):
        """LIDS inverts activity: the quieter the subject, the higher the value."""
        low, high = _lids_func(np.array([1.0])), _lids_func(np.array([1000.0]))
        assert low > high

    def test_transform_is_strictly_decreasing(self):
        values = np.array([0.0, 1.0, 5.0, 20.0, 100.0, 500.0])
        A.assert_monotonic(_lids_func(values), increasing=False, strict=True)


class TestLIDSPipeline:
    @pytest.fixture(scope="class")
    @classmethod
    def sleep_series(cls):
        return S.realistic_rest_activity(
            n_days=3, sleep_start_hour=23, sleep_duration_hours=8, n_awakenings=0
        )

    def test_transform_returns_an_aligned_series(self, sleep_series):
        lids = LIDS().lids_transform(sleep_series)
        assert isinstance(lids, pd.Series)
        assert len(lids) == len(sleep_series)

    @pytest.mark.parametrize("method", ["mva", "kernel"])
    def test_smoothing_methods_are_accepted(self, sleep_series, method):
        lids = LIDS().lids_transform(sleep_series, method=method)
        assert np.isfinite(lids.dropna()).all()

    def test_fit_populates_results_and_metrics(self, sleep_series):
        model = LIDS()
        lids = model.lids_transform(sleep_series)
        model.lids_fit(lids, scan_period=False, verbose=False)
        assert model.lids_fit_results is not None
        # lids_pearson_r returns a scipy PearsonRResult, not a bare float.
        correlation = model.lids_pearson_r(lids)
        assert np.isfinite(correlation.statistic)
        assert np.isfinite(correlation.pvalue)
        assert np.isfinite(model.lids_mri(lids))

    def test_pearson_r_is_a_correlation(self, sleep_series):
        model = LIDS()
        lids = model.lids_transform(sleep_series)
        model.lids_fit(lids, scan_period=False)
        result = model.lids_pearson_r(lids)
        A.assert_within_range(result.statistic, -1.0, 1.0, name="LIDS r")
        A.assert_within_range(result.pvalue, 0.0, 1.0, name="LIDS p")

    def test_accessing_results_before_fitting_warns(self):
        with pytest.warns(UserWarning, match="Run lids_fit"):
            assert LIDS().lids_fit_results is None

    def test_invalid_fit_function_is_rejected(self):
        with pytest.raises((ValueError, KeyError)):
            LIDS(fit_func="not-a-function")


class TestLIDSFilter:
    """``LIDS.filter`` selects sleep bouts by duration."""

    @staticmethod
    def _bouts(hours):
        return [S.flat(n_days=1, value=1.0)[: int(h * 60)] for h in hours]

    def test_keeps_only_bouts_inside_the_duration_window(self):
        kept = LIDS.filter(self._bouts([2, 5, 20]), duration_min="3H", duration_max="12H")
        assert len(kept) == 1

    def test_widening_the_window_keeps_more(self):
        bouts = self._bouts([2, 5, 20])
        assert len(LIDS.filter(bouts, duration_min="1H", duration_max="24H")) == 3

    def test_empty_input_gives_empty_output(self):
        assert LIDS.filter([], duration_min="3H", duration_max="12H") == []

    @pytest.mark.xfail(
        reason="LIDS.filter is annotated as taking a Series but needs a list of Series",
        raises=AttributeError,
        strict=True,
    )
    def test_accepts_the_annotated_type(self):
        LIDS.filter(S.flat(n_days=1), duration_min="3H", duration_max="12H")


# Fractal -- DFA / MF-DFA


@pytest.fixture(scope="module")
def dfa_scales():
    """Window sizes in **minutes** (see TestFractalUnits)."""
    return Fractal.equally_spaced_logscale_range(15, start=10, stop=500)


class TestFractalBuildingBlocks:
    def test_profile_is_the_mean_removed_cumulative_sum(self):
        values = np.array([1.0, 3.0, 2.0, 6.0])
        np.testing.assert_allclose(
            Fractal.profile(values), np.cumsum(values - values.mean())
        )

    def test_profile_of_a_constant_series_is_flat_zero(self):
        np.testing.assert_allclose(Fractal.profile(np.full(10, 5.0)), 0.0, atol=1e-12)

    @pytest.mark.parametrize("n", [3, 4, 10, 50])
    def test_non_overlapping_segmentation_tiles_the_series(self, n):
        segments = Fractal.segmentation(np.arange(100.0), n)
        assert segments.shape == (100 // n, n)

    @pytest.mark.parametrize("n", [4, 10, 50])
    def test_overlapping_segmentation_roughly_doubles_the_count(self, n):
        plain = Fractal.segmentation(np.arange(100.0), n, overlap=False)
        overlapped = Fractal.segmentation(np.arange(100.0), n, overlap=True)
        assert len(overlapped) > len(plain)

    def test_residuals_of_a_perfect_line_are_zero(self):
        assert Fractal.local_msq_residuals(np.arange(50.0), 1) == pytest.approx(0.0)

    def test_residuals_of_a_parabola_vanish_only_at_degree_two(self):
        segment = np.arange(50.0) ** 2
        assert Fractal.local_msq_residuals(segment, 1) > 1.0
        assert Fractal.local_msq_residuals(segment, 2) == pytest.approx(0.0, abs=1e-6)

    def test_logscale_range_is_strictly_increasing_and_bounded(self):
        scales = Fractal.equally_spaced_logscale_range(15, start=10, stop=500)
        A.assert_monotonic(scales, increasing=True, strict=True)
        assert scales[0] >= 10 and scales[-1] <= 500
        assert np.issubdtype(scales.dtype, np.integer)

    @pytest.mark.xfail(
        reason="Fractal.segmentation has its forward and backward branches swapped",
        strict=True,
    )
    def test_forward_segmentation_starts_at_the_beginning(self):
        first = Fractal.segmentation(np.arange(100.0), 10, backward=False)[0]
        np.testing.assert_array_equal(first[:4], [0.0, 1.0, 2.0, 3.0])

    def test_the_inversion_is_pinned(self):
        """Pin current behaviour so the xfail above cannot be misread."""
        forward = Fractal.segmentation(np.arange(100.0), 10, backward=False)[0]
        backward = Fractal.segmentation(np.arange(100.0), 10, backward=True)[0]
        np.testing.assert_array_equal(forward[:4], [90.0, 91.0, 92.0, 93.0])
        np.testing.assert_array_equal(backward[:4], [0.0, 1.0, 2.0, 3.0])


class TestDFAValidation:
    """The canonical validation of any DFA implementation."""

    def test_white_noise_gives_hurst_one_half(self, dfa_scales):
        series = S.gaussian_noise(n_days=3, mu=0.0, sigma=1.0, seed=0)
        fluctuations = Fractal.dfa(series, dfa_scales, deg=1)
        h = float(np.ravel(Fractal.generalized_hurst_exponent(fluctuations, dfa_scales))[0])
        # Uncorrelated noise: H = 0.5 in the limit, a few percent off over 3 days
        assert h == pytest.approx(0.5, abs=0.05)

    def test_brown_noise_gives_hurst_three_halves(self, dfa_scales):
        series = S.brown_noise(n_days=3, seed=0)
        fluctuations = Fractal.dfa(series, dfa_scales, deg=1)
        h = float(np.ravel(Fractal.generalized_hurst_exponent(fluctuations, dfa_scales))[0])
        assert h == pytest.approx(1.5, abs=0.05)

    def test_integrating_a_signal_raises_hurst_by_one(self, dfa_scales):
        """H(cumsum(x)) = H(x) + 1 -- the defining property of the exponent."""
        white = S.gaussian_noise(n_days=3, mu=0.0, sigma=1.0, seed=1)
        brown = S.brown_noise(n_days=3, seed=1)
        h_white = float(
            np.ravel(Fractal.generalized_hurst_exponent(Fractal.dfa(white, dfa_scales), dfa_scales))[0]
        )
        h_brown = float(
            np.ravel(Fractal.generalized_hurst_exponent(Fractal.dfa(brown, dfa_scales), dfa_scales))[0]
        )
        assert h_brown - h_white == pytest.approx(1.0, abs=0.1)

    def test_fluctuations_grow_with_scale(self, dfa_scales):
        series = S.brown_noise(n_days=3, seed=2)
        fluctuations = Fractal.dfa(series, dfa_scales, deg=1)
        assert fluctuations[-1] > fluctuations[0]
        assert np.isfinite(fluctuations).all()

    def test_dfa_is_deterministic(self, dfa_scales):
        series = S.gaussian_noise(n_days=2, seed=3)
        np.testing.assert_allclose(
            Fractal.dfa(series, dfa_scales), Fractal.dfa(series, dfa_scales)
        )


class TestMFDFA:
    @pytest.fixture(scope="class")
    @classmethod
    def q_values(cls):
        return np.array([-3.0, -1.0, 2.0, 3.0, 5.0])

    def test_output_is_scales_by_q(self, dfa_scales, q_values):
        series = S.gaussian_noise(n_days=2, seed=0)
        result = np.asarray(Fractal.mfdfa(series, dfa_scales, q_values))
        assert result.shape == (len(dfa_scales), len(q_values))

    def test_q_equals_two_reproduces_plain_dfa(self, dfa_scales, q_values):
        """MF-DFA at q=2 *is* DFA -- if these disagree, one of them is wrong."""
        series = S.gaussian_noise(n_days=2, seed=0)
        multifractal = np.asarray(Fractal.mfdfa(series, dfa_scales, q_values))
        index = int(np.flatnonzero(q_values == 2.0)[0])
        np.testing.assert_allclose(
            multifractal[:, index], Fractal.dfa(series, dfa_scales), rtol=1e-9
        )

    def test_hurst_spectrum_is_non_increasing_in_q(self, dfa_scales, q_values):
        series = S.realistic_rest_activity(n_days=3)
        multifractal = np.asarray(Fractal.mfdfa(series, dfa_scales, q_values))
        spectrum = [
            float(
                np.ravel(
                    Fractal.generalized_hurst_exponent(multifractal[:, i], dfa_scales)
                )[0]
            )
            for i in range(len(q_values))
        ]
        A.assert_monotonic(spectrum, increasing=False)

    def test_monofractal_signal_has_a_flat_spectrum(self, dfa_scales, q_values):
        """White noise is monofractal: h(q) barely varies with q."""
        series = S.gaussian_noise(n_days=3, seed=5)
        multifractal = np.asarray(Fractal.mfdfa(series, dfa_scales, q_values))
        spectrum = [
            float(
                np.ravel(
                    Fractal.generalized_hurst_exponent(multifractal[:, i], dfa_scales)
                )[0]
            )
            for i in range(len(q_values))
        ]
        assert max(spectrum) - min(spectrum) < 0.2


class TestFractalUnits:
    """``n_array`` is in minutes, not samples -- an easy and costly mistake."""

    def test_scales_are_interpreted_as_minutes(self):
        """The same physical window must give the same answer at any resolution."""
        scales = np.array([60, 120, 240, 480])
        fine = S.brown_noise(n_days=3, sampling_period=60, seed=7)
        coarse = fine.resample("2min").mean()

        h_fine = float(
            np.ravel(Fractal.generalized_hurst_exponent(Fractal.dfa(fine, scales), scales))[0]
        )
        h_coarse = float(
            np.ravel(Fractal.generalized_hurst_exponent(Fractal.dfa(coarse, scales), scales))[0]
        )
        assert h_fine == pytest.approx(h_coarse, abs=0.25)

    def test_series_without_a_frequency_is_rejected_clearly(self, dfa_scales):
        series = S.gaussian_noise(n_days=2)
        series.index.freq = None
        with pytest.raises(ValueError, match="sampling frequency"):
            Fractal.dfa(series, dfa_scales)

    @pytest.mark.xfail(
        reason="A DFA window shorter than the sampling period fails with an opaque numpy error",
        strict=True,
    )
    def test_window_below_the_sampling_period_reports_a_useful_error(self):
        series = S.gaussian_noise(n_days=2, sampling_period=300)
        with pytest.raises(ValueError, match="(?i)window|scale|sampling period"):
            Fractal.dfa(series, np.array([4, 8]))

    def test_the_opaque_failure_is_pinned(self):
        series = S.gaussian_noise(n_days=2, sampling_period=300)
        with pytest.raises(ValueError, match="slice step cannot be zero"):
            Fractal.dfa(series, np.array([4, 8]))
