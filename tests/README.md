# circStudio test suite

Automated checks for the circStudio library (the programmatic interface). The
graphical interface in `app/` is not covered.

## Running the tests

```bash
pip install pytest pytest-mock
pytest tests                    # everything
pytest tests -m "not slow"      # skip the slow reader tests
pytest tests/analysis/test_sleep.py   # one area
```

circStudio needs Python 3.12 or newer. `conftest.py` puts `src/` on the import
path, so no install step is needed.

The summary line reports:

- **passed**: behaved as expected.
- **skipped**: could not run here, usually because a bundled example file is missing.
- **xfailed**: an expected failure documenting a known bug (see [Known problems](#known-problems)).
- **failed**: something is wrong and needs attention.

## Layout

```
tests/
├── conftest.py              shared fixtures (bundled data paths, loaded recordings)
├── test_helpers.py          checks that the helpers below are themselves correct
├── helpers/
│   ├── signals.py           seeded synthetic signals with derived ground truth
│   └── assertions.py        shared assertions
├── io/
│   ├── test_readers.py      one class per file format, plus a shared reader contract
│   └── test_raw_and_mask.py the Raw container, BaseLog and Mask
├── preprocessing/
│   └── test_nonwear.py      Troiano and Choi non-wear detection
├── analysis/
│   ├── test_metrics_nonparametric.py   IS, IV, L5, M10, RA, ADAT and per-period variants
│   ├── test_rhythm_models.py           Cosinor, SSA, FLM, LIDS, fractal analysis
│   ├── test_sleep.py                   sleep scoring, summary measures, sleep diary
│   ├── test_circadian_models.py        light schedules and circadian oscillator models
│   ├── test_circular_stats.py          circular statistics
│   └── test_tools.py                   internal helpers in analysis/tools.py
└── package/
    └── test_namespace.py    public imports, exports and import hygiene
```

Within each file, tests are grouped into one class per function or behaviour.

## How correctness is decided

Tests compare circStudio against signals whose answer is known in advance, not
against a snapshot of its earlier output. `helpers/signals.py` builds synthetic
recordings and its docstrings derive their expected properties; if you change a
generator, redo the derivation. Examples:

| Synthetic recording | Expected result | Why |
|---|---|---|
| 12 h on / 12 h off square wave | L5 = off level, M10 = on level, RA = 1 | The 5 quietest hours lie in the off block, the 10 most active in the on block |
| Pure 24 h cosine | Cosinor returns its amplitude and acrophase | It is the curve the model fits |
| White noise | DFA gives H ≈ 0.5 | Textbook value for an uncorrelated process |
| Cumulative sum of that noise | H ≈ 1.5 | Integration raises the Hurst exponent by 1 |
| A night with sleep at a set time | Every scorer places sleep there | It was put there |

Where no closed form exists, tests check properties that must hold for any
input: a proportion stays in [0, 1], a decomposition sums back to the input,
scaling the counts leaves IS and IV unchanged.

The file readers are tested on the example recordings in `src/circstudio/data`.

## Known problems

A test that exposes a real defect is marked `xfail(strict=True)` with a
one-line reason. If the bug is fixed the test passes and the run fails, so the
marker has to be removed. Most have a companion test that pins the current
wrong behaviour. The full description of each defect is below.

### `io/test_readers.py`

- **`TestAWD::test_explicit_frequency_overrides_the_header`**: read_awd's `frequency` argument is silently ignored. Passing frequency='30s' for a 1-minute file still yields frequency=1min -- the header value wins and the caller is not told. A documented override that does nothing is worse than no override: anyone correcting a mis-stated header believes they have, and every downstream epoch-length calculation is then wrong.
- **`TestAGD::test_white_light_accessor_works`**: Shared root cause, 3 readers: the `white_light` property raises AttributeError: 'DataFrame' object has no attribute 'get_channel'. Every implementation reads `self.light.get_channel('whitelight')`, but `self.light` is a plain pandas DataFrame with no such method -- it was evidently once a custom channel container and the accessors were never updated. This breaks AGD, DQT and TAL identically. Light exposure is the reason to use these formats over a plain activity logger, and the documented way to reach it raises. The data itself is fine and reachable via `.light`.
- **`TestTalHangs::test_tal_terminates_on_the_mtn_file`**: Infinite hang: read_tal never returns on the bundled test_sample.mtn. It does not raise, print, or consume CPU -- it simply blocks forever, so nothing times out and nothing is logged. This matters because app/utils.py maps the .mtn extension to the candidate list ['tal', 'atr'], with TAL tried FIRST. A user uploading a .mtn file through the web interface therefore hangs their session with no error message; the fallback to the ATR reader is never reached. Either read_tal should reject a file it cannot parse, or .mtn should not be routed to it.
- **`TestTalHangs::test_an_empty_file_is_rejected`**: Infinite hang: read_tal blocks forever on an empty file. Every other reader raises. Because app/utils.py routes both .mtn and .txt uploads to the TAL reader, a user who uploads a truncated or wrong-format file through the web interface hangs their session with no error and no timeout -- the worst possible failure mode, since nothing is logged and the fallback reader is never reached.
- **`TestTalHangs::test_a_garbage_file_is_rejected`**: Same infinite hang as test_an_empty_file_is_rejected.

### `analysis/test_metrics_nonparametric.py`

- **`TestInterdailyStability::test_perfectly_repeating_pattern_gives_one`**: IS exceeds its own upper bound of 1. Witting et al. (1990) define IS = d24h/d1h with *population* variances (divide by n). The implementation uses pandas .var(), which defaults to ddof=1, for both terms -- but the two terms have different sample sizes (p = epochs per day for the numerator, N = total epochs for the denominator), so the Bessel corrections do not cancel. For a perfectly repeating D-day recording the result is (N-1)/(D*(E-1)) instead of exactly 1: 1.000596 for 7 days of 1 min data. The bias grows as the recording shortens (1.0042 for a 2-day recording), and any downstream code that assumes IS <= 1 is unsafe. Fix: pass ddof=0 to both .var() calls.
- **`TestInterdailyStability::test_lies_in_the_unit_interval`**: Same ddof bug as test_perfectly_repeating_pattern_gives_one: the square wave returns 1.0006, breaching the documented [0, 1] range.

### `analysis/test_rhythm_models.py`

- **`TestCosinorRecovery::test_acrophase_at_the_recording_start_is_recovered`**: A rhythm whose peak falls exactly at the start of the recording is reported as arrhythmic. The Acrophase parameter is bounded to [0, 2*pi] and initialised at pi. A signal peaking at t=0 has a true acrophase of exactly 0 -- on the bound. lmfit maps bounded parameters through a transform whose gradient vanishes at the bounds, so the optimiser cannot reach it; instead it drives Amplitude to 0 and Period drifts to 1442. Every other acrophase from 2 h to 22 h recovers exactly, so this is specifically a boundary failure. It matters because recording start times are arbitrary: a subject whose activity peaks when the device was fitted silently yields amplitude 0. Fix: use an unbounded acrophase and wrap the result, or initialise from the data (e.g. via the FFT phase) rather than a fixed pi.
- **`TestSSAValidation::test_window_longer_than_the_series_is_rejected`**: An embedding window longer than the series is accepted and silently produces a degenerate decomposition instead of raising. K is computed as N - L + 1, so a 500 h window on a 3-day series gives L=3000, K=-2567 -- a NEGATIVE number of columns. fit() then returns an empty spectrum (len(variance_explained) == 0) and every downstream reconstruction yields nothing, with no error at any point. The window length should be validated against the series length in __init__.
- **`TestLIDSFilter::test_accepts_the_annotated_type`**: Type annotation: LIDS.filter is annotated `ts: pd.Series` and its docstring says it filters 'data'. In fact it forwards to filter_ts_duration, which iterates its argument and calls `s.index[-1] - s.index[0]` on each element -- so it requires an *iterable of Series* (a list of sleep bouts), not a Series. Passing the annotated type iterates the Series' float values and raises AttributeError: 'float' object has no attribute 'index'. The function works, but its signature documents the one input that cannot work.
- **`TestFractalBuildingBlocks::test_forward_segmentation_starts_at_the_beginning`**: Inverted branches: Fractal.segmentation applies the wrong slice to each case: `backward=True` uses `windows[::stride]` (a forward walk) and `backward=False` uses `windows[::-stride]`. The default (backward=False) therefore returns segments starting from the END of the series, and backward=True returns them from the start. Verified: segmentation(arange(100), 10)[0] == [90, 91, 92, 93] while backward=True gives [0, 1, 2, 3]. DFA calls both and pools the results, so aggregate fluctuation values are unaffected and the H estimates below are still correct -- but any direct caller of segmentation gets the opposite of what it asked for, and the two tilings differ at the boundary when the series length is not a multiple of n.
- **`TestFractalUnits::test_window_below_the_sampling_period_reports_a_useful_error`**: Opaque failure: a window shorter than the sampling period is silently truncated to zero samples and then crashes deep inside numpy. dfa computes `factor = Timedelta('1min') / freq` and calls `int(factor * n)`. At 5 min epochs, factor = 0.2, so a 4-minute scale becomes int(0.8) = 0 samples; segmentation then does `windows[::-0]` and raises 'ValueError: slice step cannot be zero'. The user gets a numpy slicing error with no indication that the real problem is a window smaller than their epoch length. It should raise something like 'window size 4min is shorter than the sampling period 5min'.

### `analysis/test_sleep.py`

- **`TestColeKripke::test_default_settings_accept_one_minute_data`**: Usability: Cole-Kripke cannot be applied to 1-minute data with its default settings, yet every bundled example recording in src/circstudio/data is 1-minute. The default is '30sec_max_non_overlap', which requires epochs <= 30 s; the available settings are 'mean', '10sec_max_overlap', '10sec_max_non_overlap' and '30sec_max_non_overlap' -- there is no 60-second variant. So the best-known sleep algorithm in the package raises on the package's own example data unless the user happens to discover settings='mean'. Cole & Kripke (1992) do publish a 1-minute weighting, so the setting is missing rather than impossible. At minimum the default should adapt to the input sampling rate.
- **`TestSleepDiary::test_sleep_efficiency_is_a_fraction`**: SleepDiary.sleep_efficiency() returns None. Called with the recording's activity series -- exactly as its signature invites -- it produces no value and raises nothing, so anyone computing sleep efficiency across a cohort silently gets a column of None. Sleep efficiency (time asleep / time in bed) is one of the headline numbers a sleep diary exists to provide.
- **`TestSleepDiary::test_extra_states_file_loads`**: The bundled example_01_sleepdiary_extra_states.ods cannot be opened with default settings. It contains an AWAKE_IN_BED state, but the default state_index is only {ACTIVE, NAP, NIGHT, NOWEAR}, so SleepDiary raises a bare KeyError: 'AWAKE_IN_BED'. The file exists precisely to demonstrate custom states, yet nothing in the constructor signals that a matching state_index must be supplied first, and the error names the missing key rather than explaining that the state vocabulary needs extending.

### `analysis/test_circadian_models.py`

- **`TestLightCreate::test_light_phase_is_the_constant_high_level`**: Light.create does not produce the rectangular light/dark cycle its parameters describe. The dark phase is exactly `low`, but the light phase is filled with *random* intensities drawn between 0 and `high` -- values like 649, 665, 672, 576 for high=1000 -- rather than the constant `high`. Parameters named `low` and `high` unambiguously describe two levels, and every circadian entrainment protocol in the literature specifies a fixed photopic level. As it stands, two runs with identical arguments give different light input, so model output is not reproducible across calls and cannot be compared against published simulations.
- **`TestLightOperations::test_downsample_reduces_the_sample_count`**: Light.downsample(factor) raises TypeError: slice indices must be integers. The method slices `self.time_vector[::downsampling_factor]` with a value that is not coerced to int, so the documented call downsample(2) fails outright. Downsampling is the natural way to match a light schedule to a coarser recording, so this blocks a routine workflow.
- **`TestModelValidation::test_mismatched_input_and_time_lengths_are_rejected`**: Silent acceptance: Forger accepts a light vector and a time vector of different lengths and integrates anyway, without warning. The light input and the integration grid then disagree, so the model is driven by the wrong illuminance at every step and returns a plausible-looking but meaningless trajectory. Mismatched inputs are an easy mistake when a light schedule is built separately from a recording, and nothing here catches it.
- **`TestModelValidation::test_empty_light_input_is_rejected`**: Silent acceptance: an empty light input is accepted and integrated without error, yielding an empty trajectory rather than a diagnostic. Combined with the mismatched-length case above, the models perform no input validation at all.

### `analysis/test_circular_stats.py`

- **`TestCircularMedian::test_odd_length_input_works`**: NameError: in the odd-length branch, circular_median returns `signal[np.argmin(dist)]` but no name `signal` exists in that scope -- it should be `angle_vector`. Every odd-length input therefore raises NameError. The even-length branch is exercised by the module's own __main__ demo, which is presumably why this went unnoticed: that demo passes 10 angles.
- **`TestPermutationTest::test_permutation_test_runs_at_all`**: NameError: Cir_Inference_Stats.permutation_test calls `cds.circular_variance(...)` in two places, but `cds` is not defined anywhere in the module -- it should be `cls` (the method is a classmethod inheriting circular_variance from Cir_Descriptive_Stats). The function therefore raises NameError on every call and has never been executable. It is exported through `from .models.tools import *` into the top-level circstudio namespace, so this is a public API that cannot be used at all.
- **`TestPermutationTest::test_p_values_lie_in_the_unit_interval`**: Blocked by the same NameError as test_permutation_test_runs_at_all.
- **`TestPermutationTest::test_concentrated_group_separates_from_uniform_group`**: Blocked by the same NameError as test_permutation_test_runs_at_all.
- **`TestPermutationTest::test_results_are_reproducible`**: Blocked by the same NameError. Note also that permutation_test calls np.random.default_rng() with no seed, so even once the NameError is fixed the function is not reproducible and cannot be tested for exact output. It should accept a seed/rng argument.

### `analysis/test_tools.py`

- **`TestTdFormat::test_durations_over_a_day_keep_their_days`**: _td_format uses td.components.hours, which is the hour part *within* a day and therefore silently discards whole days. A duration of 25h30m formats as '01:30:00', indistinguishable from 1h30m. This matters for TAT/VAT and total bed/nap time, which can legitimately exceed 24 h across a multi-day recording.
- **`TestLightExposure::test_time_window_restricts_to_the_requested_hours`**: _light_exposure calls Series.between_time(include_end=False). The include_start/include_end arguments were deprecated in pandas 1.4 and REMOVED in pandas 2.0 in favour of `inclusive=`. The project requires pandas>=2.3.3, so any call supplying both start_time and stop_time raises TypeError. This code path is therefore dead in every supported pandas version.
- **`TestCreateInactivityMask::test_entirely_zero_signal_is_entirely_masked`**: A recording that is zero everywhere is reported as 100% valid. _create_inactivity_mask derives transitions from np.diff(binary_data); for an all-zero signal there are no transitions, so the early-return `if all(e == 0 for e in edges)` hands back an all-ones (all-valid) mask. The same early return also mishandles a recording that is entirely *above* threshold, but that case is harmless. A dead device, or a subject who never wore it, is exactly the input a non-wear mask exists to catch, and it is the one input that silently passes.

### `package/test_namespace.py`

- **`TestNamespacePollution::test_no_third_party_modules_leak_into_the_namespace`**: Namespace pollution: circstudio re-exports its dependencies. `circstudio/__init__.py` does `from .analysis import *`, and `analysis/__init__.py` does `from .metrics.metrics import *` etc. None of those submodules define __all__, so the star-import also copies every module they imported. The result is that `circstudio.np`, `circstudio.pd`, `circstudio.os`, `circstudio.re`, `circstudio.plt`, `circstudio.go`, `circstudio.sns`, `circstudio.pg`, `circstudio.stats` and `circstudio.warnings` are all live attributes of the public package. That freezes implementation details into the API (a user can write `circstudio.pd` and it works), and it means `from circstudio import *` clobbers the caller's own `np`/`pd`/`os`. Fix: add an explicit __all__ to each starred submodule.
- **`TestImportHygiene::test_sources_compile_without_syntax_warnings`**: Two docstrings contain LaTeX escape sequences but are not raw strings, so compiling them emits SyntaxWarning: `'\sum'` in the SleepRegularityIndex docstring (`analysis/sleep/sleep.py`) and `'\pi'` in the sleep_midpoint docstring (`analysis/sleep/scoring/smp.py`). Both are one-character fixes (prefix the docstring with r). Python has announced that invalid escape sequences will become a SyntaxError in a future release, so this is a forward-compatibility problem rather than mere noise. Note the warnings are only visible on a cold compile -- once __pycache__ is populated they vanish, which is why they have gone unnoticed.

## Adding a test

1. Put it in the folder and file matching the module under test.
2. Prefer, in order: an answer derived on paper; a property that must always
   hold; agreement between two parts of circStudio that compute the same thing.
3. New synthetic signals go in `helpers/signals.py`, with their ground truth in
   the docstring.
4. Break the function on purpose and confirm the test fails.
5. Keep comments and docstrings to one line.

## Notes

- Tests import `circstudio`, never `src.circstudio`.
- `raw_awd` is shared across the session and must not be mutated; use
  `raw_awd_fresh` for anything that filters or masks.
- Markers: `slow` (more than a few seconds) and `needs_data` (reads a bundled
  file, skips if absent), declared in `pyproject.toml` and `conftest.py`.
- GitHub Actions runs the whole suite, slow tests included, on Ubuntu with
  Python 3.12 (`.github/workflows/tests.yml`).
- `sleep_tests.py` from the previous suite never ran (its name did not match
  pytest's pattern), so sleep-onset latency is still untested.
