from datetime import date

import pandas as pd
import pytest

from alphaflow.core.utils.storage import StoragePaths
from alphaflow.data_dumper.runner import DataDumperRunner

MARKET = "HK"
DAY = date(2024, 1, 15)


@pytest.fixture
def runner(system_config, fake_desktool, minute_bar_frame):
    """A dumper whose kdb extractor is fed by a fake desktool."""
    instance = DataDumperRunner(system_config)
    frame = minute_bar_frame([DAY])
    instance.extractors[0].client._dt = fake_desktool(kdb_result=frame)
    return instance


class TestDataDumperPipeline:
    """kdb/S3 -> raw_data CSVs, the input contract for signal_builder."""

    def test_a_single_source_lands_at_the_spec_path(self, runner, isolated_shared_drive):
        written = runner.run(market=MARKET, date="20240115", source="minute_bar")
        assert written == {"minute_bar": ["20240115"]}
        assert (isolated_shared_drive / "raw_data" / MARKET / "minute_bar" / "20240115.csv").is_file()

    def test_the_csv_is_ric_keyed_and_readable_by_signal_builder(self, runner, system_config):
        runner.run(market=MARKET, date="20240115", source="minute_bar")
        path = StoragePaths(system_config.storage).raw_data(MARKET, "minute_bar", DAY)
        frame = pd.read_csv(path)
        assert frame.columns[0] == "RIC" and not frame.empty

    def test_a_date_range_writes_one_file_per_day(self, runner):
        written = runner.run(market=MARKET, start="20240115", end="20240117", source="minute_bar")
        assert written["minute_bar"] == ["20240115", "20240116", "20240117"]

    def test_only_universe_rics_are_requested(self, runner):
        runner.run(market=MARKET, date="20240115", source="minute_bar")
        requested = runner.extractors[0].client._dt.calls[0][1]["rics"]
        assert set(requested) == set(runner.load_universe(MARKET))

    def test_the_bbg_placeholder_does_not_abort_a_full_dump(self, runner):
        written = runner.run(market=MARKET, date="20240115")
        assert written["bbg"] == [] and written["minute_bar"] == ["20240115"]
