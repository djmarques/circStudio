"""File readers for every supported device format."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from circstudio.io import (
    Raw,
    read_agd,
    read_atr,
    read_awd,
    read_dqt,
    read_mesa,
    read_rpx,
    read_tal,
)
from helpers import assertions as A

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "src" / "circstudio" / "data"

# Bundled Actiwatch variants: expected model string and whether it records light
AWD_DIALECTS = [
    ("test_sample_aw4.AWD", "Actiwatch-4", False, "1min"),
    ("test_sample_aw7.AWD", "Actiwatch-7", True, "15s"),
    ("test_sample_awi.AWD", "Actiwatch-Insomnia (pressure sens.)", False, "1min"),
    ("test_sample_awl.AWD", "Actiwatch-L (amb. light)", True, "1min"),
    ("test_sample_awlp.AWD", "Actiwatch-L-Plus (amb. light)", True, "1min"),
    ("test_sample_awmk2.AWD", "Actiwatch-Mini", False, "30s"),
    ("test_sample_aws.AWD", "Actiwatch-S (env. sound)", False, "1min"),
    ("test_sample_awt.AWD", "Actiwatch-T (temp.)", False, "1min"),
]


def data_file(name: str) -> str:
    path = DATA / name
    if not path.exists():
        pytest.skip(f"bundled data file not available: {name}")
    return str(path)


# AWD -- Actiwatch


@pytest.mark.needs_data
class TestAWD:
    def test_example_recording_header(self, awd_path):
        """Values carried over from the legacy ``tests/test_awd.py``."""
        raw = read_awd(str(awd_path))
        assert raw.frequency == pd.Timedelta(1, unit="m")
        assert raw.activity.index[0] == pd.Timestamp("1918-01-23 13:58:00")
        assert len(raw.activity) == 18401

    def test_activity_is_a_valid_series(self, awd_path):
        A.assert_valid_activity_series(read_awd(str(awd_path)).activity)

    def test_counts_are_non_negative(self, awd_path):
        assert (read_awd(str(awd_path)).activity >= 0).all()

    @pytest.mark.parametrize(
        "filename,model,has_light,freq",
        AWD_DIALECTS,
        ids=[d[0].replace("test_sample_", "").replace(".AWD", "") for d in AWD_DIALECTS],
    )
    def test_every_actiwatch_model_parses(self, filename, model, has_light, freq):
        """Each Actiwatch generation writes a slightly different header."""
        raw = read_awd(data_file(filename))
        assert raw.model == model
        assert (raw.light is not None) == has_light
        assert raw.frequency == pd.Timedelta(freq)
        assert len(raw.activity) > 0

    def test_engine_option_does_not_change_the_data(self):
        """``engine='c'`` and ``engine='python'`` must agree exactly."""
        path = data_file("test_sample_aw4.AWD")
        pandas_engine = read_awd(path, engine="python").activity
        c_engine = read_awd(path, engine="c").activity
        pd.testing.assert_series_equal(pandas_engine, c_engine)

    @pytest.mark.xfail(reason="read_awd ignores the frequency argument", strict=True)
    def test_explicit_frequency_overrides_the_header(self, awd_path):
        raw = read_awd(str(awd_path), frequency="30s")
        assert raw.frequency == pd.Timedelta(30, unit="s")

    def test_the_ignored_frequency_argument_is_pinned(self, awd_path):
        assert read_awd(str(awd_path), frequency="30s").frequency == pd.Timedelta(
            1, unit="m"
        )

    def test_truncated_file_raises(self, tmp_path):
        broken = tmp_path / "truncated.AWD"
        broken.write_text("only one line\n")
        with pytest.raises(Exception):
            read_awd(str(broken))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            read_awd(str(tmp_path / "absent.AWD"))


# ATR -- ActTrust (Condor Instruments)


@pytest.mark.needs_data
class TestATR:
    def test_example_recording_header(self, atr_path):
        """Values carried over from the legacy ``tests/test_atr.py``."""
        raw = read_atr(str(atr_path))
        assert raw.activity.index[0] == pd.Timestamp("1918-01-01 09:00:00")
        assert len(raw.activity) == 4 * 1440, "a 4-day recording at 1 min epochs"
        assert raw.frequency == pd.Timedelta(1, unit="m")

    def test_carries_both_activity_and_light(self, atr_path):
        raw = read_atr(str(atr_path))
        A.assert_valid_activity_series(raw.activity)
        assert raw.light is not None
        assert len(raw.light) == len(raw.activity)

    @pytest.mark.parametrize("mode", ["PIM", "TAT", "ZCM"])
    def test_activity_modes_are_selectable(self, atr_path, mode):
        """ActTrust records three activity measures; each must be readable."""
        raw = read_atr(str(atr_path), activity_mode=mode)
        assert len(raw.activity) == 4 * 1440
        assert raw.activity.name == mode

    def test_default_activity_mode_is_pim(self, atr_path):
        assert read_atr(str(atr_path)).activity.name == "PIM"

    def test_a_wrong_skip_rows_fails_loudly(self, atr_path):
        """Good behaviour: skipping into the middle of the header is rejected."""
        with pytest.raises(ValueError, match="usual header"):
            read_atr(str(atr_path), skip_rows=5)


# AGD -- ActiGraph


@pytest.mark.needs_data
class TestAGD:
    @pytest.fixture(scope="class")
    @classmethod
    def raw(cls, agd_path):
        return read_agd(str(agd_path))

    def test_reads_a_ten_second_recording(self, raw):
        assert raw.frequency == pd.Timedelta(10, unit="s")
        assert len(raw.activity) == 5394
        assert raw.activity.index[0] == pd.Timestamp("2019-04-15 15:00:00")

    def test_light_data_is_present(self, raw):
        assert raw.light is not None
        assert len(raw.light) == len(raw.activity)

    @pytest.mark.xfail(
        reason="white_light calls get_channel on a plain DataFrame (AGD, DQT and TAL)",
        raises=AttributeError,
        strict=True,
    )
    def test_white_light_accessor_works(self, raw):
        assert raw.white_light is not None

    def test_the_white_light_failure_is_pinned(self, raw):
        with pytest.raises(AttributeError, match="get_channel"):
            raw.white_light

    def test_inclinometer_channels_are_aligned_and_distinct(self, raw):
        """Four separate posture channels, each one value per epoch."""
        channels = {}
        for name in ("incline_off", "incline_standing", "incline_sitting", "incline_lying"):
            channel = getattr(raw, name)
            assert channel is not None, f"{name} is missing"
            assert len(channel) == len(raw.activity)
            assert (channel >= 0).all(), f"{name} holds a negative duration"
            channels[name] = channel

        reference = channels["incline_off"]
        assert not all(reference.equals(c) for c in channels.values()), (
            "the four posture channels must not all be the same column"
        )

    def test_incline_position_is_a_method_returning_one_value_per_epoch(self, raw):
        """``incline_position`` reads like a property but must be called."""
        position = raw.incline_position()
        assert len(position) == len(raw.activity)

    def test_a_non_sqlite_file_is_rejected(self, tmp_path):
        fake = tmp_path / "fake.agd"
        fake.write_text("not a database")
        with pytest.raises(Exception):
            read_agd(str(fake))


# DQT -- Daqtometer


@pytest.mark.needs_data
class TestDQT:
    def test_reads_the_csv_export(self):
        raw = read_dqt(data_file("test_sample_dqt.csv"))
        assert raw.frequency == pd.Timedelta(1, unit="s")
        assert len(raw.activity) == 43200, "12 h at 1 s resolution"
        A.assert_valid_activity_series(raw.activity)

    def test_light_data_is_present(self):
        raw = read_dqt(data_file("test_sample_dqt.csv"))
        assert raw.light is not None

    def test_white_light_accessor_fails_the_same_way_as_agd(self):
        """The identical stale-accessor bug -- see TestAGD.test_white_light_accessor_works."""
        raw = read_dqt(data_file("test_sample_dqt.csv"))
        with pytest.raises(AttributeError, match="get_channel"):
            raw.white_light

    def test_wrong_header_size_fails_loudly(self):
        with pytest.raises(Exception):
            read_dqt(data_file("test_sample_dqt.csv"), header_size=0)


# TAL -- Tempatilumi


@pytest.mark.needs_data
class TestTAL:
    @pytest.fixture(scope="class")
    @classmethod
    def raw(cls):
        return read_tal(data_file("test_sample_tal.txt"))

    def test_reads_a_week_at_one_minute(self, raw):
        assert raw.frequency == pd.Timedelta(1, unit="m")
        assert len(raw.activity) == 10080, "exactly 7 days at 1 min epochs"
        assert raw.activity.index[0] == pd.Timestamp("2000-01-01 00:00:00")

    def test_white_light_accessor_fails_the_same_way_as_agd(self, raw):
        """The identical stale-accessor bug -- see TestAGD."""
        with pytest.raises(AttributeError, match="get_channel"):
            raw.white_light

    def test_temperature_is_absent_from_this_example(self, raw):
        """Unlike white_light, temperature returns None cleanly."""
        assert raw.temperature is None

    def test_exposes_events(self, raw):
        assert raw.events is not None
        assert len(raw.events) > 0


class TestTalHangs:
    """``read_tal`` blocks forever instead of rejecting a file it cannot parse."""

    @staticmethod
    def outcome(reader: str, path: Path, timeout: int = 8) -> str:
        """Run the reader in a subprocess and return 'ok', 'raises' or 'hang'."""
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from circstudio.io import %s as reader\n"
            "reader(%r)\n" % (str(PROJECT_ROOT / "src"), reader, str(path))
        )
        try:
            done = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return "hang"
        return "ok" if done.returncode == 0 else "raises"

    def test_a_nonexistent_path_is_rejected_normally(self, tmp_path):
        assert self.outcome("read_tal", tmp_path / "absent.txt") == "raises"

    @pytest.mark.needs_data
    @pytest.mark.slow
    @pytest.mark.xfail(reason="read_tal hangs forever on the bundled .mtn file", strict=True)
    def test_tal_terminates_on_the_mtn_file(self):
        assert self.outcome("read_tal", DATA / "test_sample.mtn") != "hang"

    @pytest.mark.needs_data
    @pytest.mark.slow
    def test_the_mtn_hang_is_pinned(self):
        assert self.outcome("read_tal", DATA / "test_sample.mtn") == "hang"

    @pytest.mark.needs_data
    def test_the_atr_fallback_for_mtn_raises(self):
        """The second candidate reader for .mtn errors rather than hanging."""
        assert self.outcome("read_atr", DATA / "test_sample.mtn") == "raises"

    @pytest.mark.slow
    @pytest.mark.xfail(reason="read_tal hangs forever on an empty file", strict=True)
    def test_an_empty_file_is_rejected(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        assert self.outcome("read_tal", empty) == "raises"

    @pytest.mark.slow
    @pytest.mark.xfail(reason="read_tal hangs forever on a garbage file", strict=True)
    def test_a_garbage_file_is_rejected(self, tmp_path):
        junk = tmp_path / "junk.txt"
        junk.write_text("lorem ipsum dolor sit amet\n" * 50)
        assert self.outcome("read_tal", junk) == "raises"

    @pytest.mark.slow
    def test_the_empty_file_hang_is_pinned(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        assert self.outcome("read_tal", empty) == "hang"


# RPX -- Respironics Actiwatch


@pytest.mark.needs_data
@pytest.mark.slow
class TestRPX:
    @pytest.fixture(scope="class")
    @classmethod
    def raw(cls):
        return read_rpx(data_file("test_sample_rpx_eng.csv"))

    def test_reads_a_thirty_second_recording(self, raw):
        assert raw.frequency == pd.Timedelta(30, unit="s")
        assert len(raw.activity) == 616320
        assert raw.activity.index[0] == pd.Timestamp("2015-04-07 09:45:00")

    def test_reports_its_language(self, raw):
        assert raw.language is not None

    def test_status_channels_are_aligned(self, raw):
        for name in ("off_wrist", "sleep_wake", "mobility", "interval_status"):
            channel = getattr(raw, name, None)
            if channel is not None:
                assert len(channel) == len(raw.activity), f"{name} is misaligned"

    def test_unknown_language_is_rejected(self):
        with pytest.raises(Exception):
            read_rpx(data_file("test_sample_rpx_eng.csv"), language="KLINGON")


@pytest.mark.needs_data
class TestRPXTranslations:
    """The header dictionaries must cover every declared language identically."""

    def test_all_languages_declare_the_same_keys(self):
        """A key missing from one translation is a silent production bug."""
        from circstudio.io.rpx import multilang

        # ``fields`` maps a language code to that language's header dictionary.
        catalogue = getattr(multilang, "fields", None)
        if not isinstance(catalogue, dict) or len(catalogue) < 2:
            pytest.skip("no multi-language header catalogue to compare")

        key_sets = {
            language: set(headers)
            for language, headers in catalogue.items()
            if isinstance(headers, dict)
        }
        if len(key_sets) < 2:
            pytest.skip("fewer than two language dictionaries to compare")
        reference_name, reference_keys = next(iter(key_sets.items()))
        for name, keys in key_sets.items():
            assert keys == reference_keys, (
                f"language dictionary '{name}' differs from '{reference_name}': "
                f"missing {sorted(reference_keys - keys)}, "
                f"extra {sorted(keys - reference_keys)}"
            )


# MESA


@pytest.mark.needs_data
class TestMESA:
    def test_reads_the_bundled_sample(self):
        raw = read_mesa(data_file("test_sample_mesa.csv"))
        assert len(raw.activity) > 0

    def test_summary_json_is_not_a_valid_input(self):
        """``sample-summary.json`` is not something ``read_mesa`` can open."""
        with pytest.raises(Exception):
            read_mesa(data_file("sample-summary.json"))

    def test_timeseries_export_is_not_a_valid_input(self):
        with pytest.raises(ValueError, match="Index line invalid"):
            read_mesa(data_file("sample-timeSeries.csv.gz"))

    @pytest.mark.parametrize(
        "filename", ["sample-summary-no-calibration.json", "sample-summary-wrong-name.json"]
    )
    def test_malformed_summaries_raise(self, filename):
        """These two files exist specifically to exercise error paths."""
        with pytest.raises(Exception):
            read_mesa(data_file(filename))


# The shared reader contract


READER_CASES = [
    ("awd", read_awd, "example_01.AWD", {}),
    ("atr", read_atr, "test_sample_atr.txt", {}),
    ("agd", read_agd, "test_sample.agd", {}),
    ("dqt", read_dqt, "test_sample_dqt.csv", {}),
    ("tal", read_tal, "test_sample_tal.txt", {}),
]


@pytest.mark.needs_data
class TestReaderContract:
    """One uniform contract every reader must satisfy."""

    @pytest.fixture(params=READER_CASES, ids=[c[0] for c in READER_CASES])
    def raw(self, request):
        _, reader, filename, kwargs = request.param
        return reader(data_file(filename), **kwargs)

    def test_returns_a_raw_instance(self, raw):
        assert isinstance(raw, Raw)

    def test_activity_is_a_regular_numeric_series(self, raw):
        A.assert_valid_activity_series(raw.activity)

    def test_frequency_is_a_positive_timedelta(self, raw):
        assert isinstance(raw.frequency, pd.Timedelta)
        assert raw.frequency > pd.Timedelta(0)

    def test_index_spacing_matches_the_declared_frequency(self, raw):
        """The header epoch length must match the timestamp spacing."""
        actual = raw.activity.index[1] - raw.activity.index[0]
        assert actual == raw.frequency

    def test_duration_is_consistent_with_length_and_frequency(self, raw):
        span = raw.activity.index[-1] - raw.activity.index[0]
        assert span == (len(raw.activity) - 1) * raw.frequency

    def test_light_when_present_is_aligned_with_activity(self, raw):
        if raw.light is not None:
            assert len(raw.light) == len(raw.activity)
            pd.testing.assert_index_equal(raw.light.index, raw.activity.index)

    def test_survives_default_filtering(self, raw):
        """Every reader's output must be usable by the Mask machinery."""
        raw.apply_filters()
        A.assert_valid_activity_series(raw.activity)


@pytest.mark.parametrize(
    "reader",
    [read_awd, read_atr, read_agd, read_dqt, read_mesa],
    ids=lambda f: f.__name__,
)
class TestReaderErrorPaths:
    """Malformed input must fail, not be silently accepted."""

    def test_nonexistent_path_raises(self, reader, tmp_path):
        with pytest.raises(Exception):
            reader(str(tmp_path / "does-not-exist.dat"))

    def test_empty_file_raises(self, reader, tmp_path):
        empty = tmp_path / "empty.dat"
        empty.write_text("")
        with pytest.raises(Exception):
            reader(str(empty))

    def test_garbage_content_raises(self, reader, tmp_path):
        junk = tmp_path / "junk.dat"
        junk.write_text("lorem ipsum dolor sit amet\n" * 50)
        with pytest.raises(Exception):
            reader(str(junk))
