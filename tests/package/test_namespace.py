"""Package namespace, exports and import hygiene."""

import importlib
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import circstudio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

IO_SYMBOLS = [
    "Raw",
    "read_atr",
    "read_awd",
    "read_agd",
    "read_dqt",
    "read_mesa",
    "read_rpx",
    "read_tal",
]

PREPROCESSING_SYMBOLS = ["detect_nonwear_troiano", "detect_nonwear_choi"]

ANALYSIS_SYMBOLS = [
    # metrics
    "daily_profile", "daily_profile_auc", "adat", "adatp", "l5", "m10", "ra",
    "l5p", "m10p", "rap", "IS", "ISp", "IV", "IVp", "summary_stats",
    "light_exposure", "TAT", "TATp", "VAT", "get_time_barycentre", "mlit",
    "mlitp", "get_extremum", "lmx", "temporal_centroid", "spectral_centroid",
    "pRA", "pAR", "kRA", "kAR",
    # feature screening
    "FeatureScreening",
    # cosinor
    "Cosinor",
    # sleep scoring
    "AonT", "AoffT", "Cole_Kripke", "Sadeh", "Scripps", "Oakley", "CSM", "SoD",
    "fSoD", "Crespo", "Crespo_AoT", "Roenneberg", "Roenneberg_AoT",
    # sleep summaries
    "SleepProfile", "SleepRegularityIndex", "SleepMidPoint", "sleep_bouts",
    "active_bouts", "sleep_durations", "active_durations", "main_sleep_bouts",
    "waso",
    # diary
    "SleepDiary",
    # decomposition / modelling
    "SSA", "FLM", "LIDS", "Fractal",
    # circadian models
    "Model", "Forger", "Jewett", "HannaySP", "HannayTP", "Hilaire07",
    "Breslow13", "Skeldon23", "ESRI", "ModelComparer",
    # light + circular stats
    "Light", "Tools", "Cir_Descriptive_Stats", "Cir_Inference_Stats",
]


class TestImportSurface:
    @pytest.mark.parametrize("name", IO_SYMBOLS)
    def test_io_symbols_are_exported(self, name):
        assert hasattr(circstudio.io, name), f"circstudio.io.{name} is missing"

    @pytest.mark.parametrize("name", PREPROCESSING_SYMBOLS)
    def test_preprocessing_symbols_are_exported(self, name):
        assert hasattr(circstudio.preprocessing, name)
        assert hasattr(circstudio, name), (
            f"{name} is re-exported at top level by circstudio/__init__.py"
        )

    @pytest.mark.parametrize("name", ANALYSIS_SYMBOLS)
    def test_analysis_symbols_reach_the_top_level(self, name):
        assert hasattr(circstudio.analysis, name), f"circstudio.analysis.{name} missing"
        assert hasattr(circstudio, name), f"circstudio.{name} missing"

    def test_subpackages_are_importable(self):
        for sub in ("io", "analysis", "preprocessing"):
            assert hasattr(circstudio, sub)
            importlib.import_module(f"circstudio.{sub}")


class TestDunderAll:
    def test_io_all_matches_reality(self):
        for name in circstudio.io.__all__:
            assert hasattr(circstudio.io, name), (
                f"circstudio.io.__all__ advertises '{name}' which does not exist"
            )

    def test_io_all_covers_every_reader(self):
        assert set(circstudio.io.__all__) == set(IO_SYMBOLS)

    def test_preprocessing_all_matches_reality(self):
        for name in circstudio.preprocessing.__all__:
            assert hasattr(circstudio.preprocessing, name)

    def test_analysis_tools_all_matches_reality(self):
        from circstudio.analysis import tools

        for name in tools.__all__:
            assert hasattr(tools, name), f"analysis.tools.__all__ advertises '{name}'"


class TestNamespacePollution:
    def test_no_builtins_are_shadowed(self):
        """A star-import that shadows a builtin is a latent trap."""
        shadowed = {
            name
            for name in dir(circstudio)
            if not name.startswith("_") and name in dir(__builtins__)
        }
        assert not shadowed, f"circstudio shadows builtins: {sorted(shadowed)}"

    @pytest.mark.xfail(
        reason="Star imports without __all__ leak np, pd, os and others into circstudio",
        strict=True,
    )
    def test_no_third_party_modules_leak_into_the_namespace(self):
        """``from .x import *`` without ``__all__`` drags in imported modules."""
        import types

        leaked = {
            name
            for name in dir(circstudio)
            if not name.startswith("_")
            and isinstance(getattr(circstudio, name), types.ModuleType)
            and getattr(circstudio, name).__name__.split(".")[0]
            not in {"circstudio"}
        }
        assert not leaked, (
            f"third-party modules exposed as circstudio attributes: {sorted(leaked)}"
        )

    def test_duplicated_helper_names_across_subpackages_are_distinguishable(self):
        """``_window_convolution`` is defined twice, in two different modules."""
        csm_module = importlib.import_module("circstudio.analysis.sleep.scoring.csm")
        sleep_tools = importlib.import_module("circstudio.analysis.sleep.sleep_tools")

        assert hasattr(csm_module, "_window_convolution")
        assert hasattr(sleep_tools, "_window_convolution")
        # They are genuinely distinct objects, not an accidental re-export.
        assert csm_module._window_convolution is not sleep_tools._window_convolution

    def test_three_modules_named_tools_do_not_collide(self):
        """Both ``tools`` modules and the ``Tools`` class stay reachable and distinct."""
        from circstudio.analysis import tools as analysis_tools
        from circstudio.analysis.models import tools as model_tools

        assert analysis_tools is not model_tools
        assert hasattr(analysis_tools, "_binarize")
        assert hasattr(model_tools, "Tools")


class TestImportHygiene:
    def test_import_has_no_side_effects_on_stdout(self):
        """Importing must not print. A stray print in a library is a real bug."""
        result = subprocess.run(
            [sys.executable, "-c", "import circstudio"],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(SRC_DIR), "PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "", f"import printed: {result.stdout!r}"

    def test_import_does_not_require_a_display(self):
        """No GUI matplotlib backend may be selected at import time."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import circstudio, matplotlib; print(matplotlib.get_backend())",
            ],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(SRC_DIR), "PATH": "/usr/bin:/bin", "MPLBACKEND": "Agg"},
        )
        assert result.returncode == 0, result.stderr
        assert "agg" in result.stdout.lower()

    @pytest.mark.xfail(
        reason="Two docstrings with LaTeX escapes are not raw strings and emit SyntaxWarning",
        strict=True,
    )
    def test_sources_compile_without_syntax_warnings(self):
        offenders = self._syntax_warning_offenders()
        assert not offenders, "invalid escape sequences in: " + ", ".join(offenders)

    def test_the_syntax_warnings_are_exactly_the_two_known_ones(self):
        """Pin the set so a third one cannot creep in unnoticed."""
        offenders = self._syntax_warning_offenders()
        assert sorted(offenders) == [
            "analysis/sleep/scoring/smp.py",
            "analysis/sleep/sleep.py",
        ], f"unexpected set of invalid-escape modules: {sorted(offenders)}"

    @staticmethod
    def _syntax_warning_offenders():
        """Compile every source file and collect those emitting SyntaxWarning."""
        package_root = SRC_DIR / "circstudio"
        offenders = []
        for path in sorted(package_root.rglob("*.py")):
            source = path.read_text(encoding="utf-8", errors="ignore")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", SyntaxWarning)
                try:
                    compile(source, str(path), "exec")
                except SyntaxError:
                    continue
            if any(issubclass(w.category, SyntaxWarning) for w in caught):
                offenders.append(str(path.relative_to(package_root)))
        return offenders

    def test_repeated_import_is_stable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            importlib.reload(importlib.import_module("circstudio.io"))


class TestPackagedData:
    def test_data_directory_is_present(self):
        data_dir = SRC_DIR / "circstudio" / "data"
        assert data_dir.is_dir(), (
            "pyproject declares source-include = ['src/circstudio/data/**'] but "
            "the directory is missing"
        )

    def test_canonical_example_files_ship(self):
        data_dir = SRC_DIR / "circstudio" / "data"
        for name in ("example_01.AWD", "test_sample.agd", "sample-summary.json"):
            assert (data_dir / name).exists(), f"bundled data file {name} is missing"

    def test_data_directory_is_not_importable_as_a_package(self):
        """It ships as package *data*, so it must not need an __init__.py."""
        data_dir = SRC_DIR / "circstudio" / "data"
        assert not (data_dir / "__init__.py").exists()


class TestVersionMetadata:
    def test_version_matches_pyproject(self):
        import tomllib

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as fh:
            declared = tomllib.load(fh)["project"]["version"]

        module_version = getattr(circstudio, "__version__", None)
        if module_version is None:
            pytest.skip("circstudio does not expose __version__")
        assert module_version == declared

    def test_pyproject_declares_the_supported_python(self):
        import tomllib

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as fh:
            project = tomllib.load(fh)["project"]
        assert project["requires-python"] == ">=3.12"
