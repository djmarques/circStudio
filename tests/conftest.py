"""Shared pytest configuration and fixtures for the circStudio test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Put src/ and tests/ on the import path before anything imports circstudio
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = SRC_DIR / "circstudio" / "data"

for _path in (SRC_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Headless plotting; must precede any pyplot import
import matplotlib  # noqa: E402

matplotlib.use("Agg")


def pytest_configure(config):
    """Register markers here as well as in pyproject.toml."""
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
    config.addinivalue_line("markers", "needs_data: requires a bundled file in src/circstudio/data")


def _data_file(name: str) -> Path:
    """Resolve a bundled data file, skipping the test if it is absent."""
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"bundled data file not available: {name}")
    return path


# Bundled data files


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not DATA_DIR.is_dir():
        pytest.skip("src/circstudio/data is not available")
    return DATA_DIR


@pytest.fixture(scope="session")
def awd_path() -> Path:
    """The example recording used across the docs and tutorials."""
    return _data_file("example_01.AWD")


@pytest.fixture(scope="session")
def atr_path() -> Path:
    return _data_file("test_sample_atr.txt")


@pytest.fixture(scope="session")
def agd_path() -> Path:
    return _data_file("test_sample.agd")


@pytest.fixture(scope="session")
def sleep_diary_ods() -> Path:
    return _data_file("example_01_sleepdiary.ods")


@pytest.fixture(scope="session")
def sleep_diary_extra_states_ods() -> Path:
    """Diary containing states beyond the default NIGHT/NAP/NOWEAR set."""
    return _data_file("example_01_sleepdiary_extra_states.ods")


@pytest.fixture(scope="session")
def mask_log_csv() -> Path:
    return _data_file("example_masklog.csv")


@pytest.fixture(scope="session")
def sst_logs() -> dict[str, Path]:
    """The same start/stop-time log in four file formats."""
    names = {
        "csv": "example_sstlog.csv",
        "ods": "example_sstlog.ods",
        "xls": "example_sstlog.xls",
        "xlsx": "example_sstlog.xlsx",
    }
    return {k: DATA_DIR / v for k, v in names.items() if (DATA_DIR / v).exists()}


# Loaded recordings


@pytest.fixture(scope="session")
def raw_awd(awd_path):
    """A loaded ``Raw`` from ``example_01.AWD``; do not mutate."""
    from circstudio.io import read_awd

    return read_awd(str(awd_path))


@pytest.fixture
def raw_awd_fresh(awd_path):
    """A freshly loaded ``Raw``, safe to mutate."""
    from circstudio.io import read_awd

    return read_awd(str(awd_path))


@pytest.fixture
def synthetic_raw():
    """A ``Raw`` built from a synthetic square wave with known properties."""
    import pandas as pd

    from circstudio.io import Raw
    from helpers import signals

    activity = signals.squarewave(n_days=7)
    light = signals.light_squarewave(n_days=7)
    frequency = pd.Timedelta(signals.DEFAULT_SAMPLING_PERIOD, unit="s")
    df = pd.DataFrame({"activity": activity, "light": light})
    return Raw(
        df=df,
        period=frequency * len(activity),
        frequency=frequency,
        activity=activity,
        light=light,
        start_time=activity.index[0],
    )
