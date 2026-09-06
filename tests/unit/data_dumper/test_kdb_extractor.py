from datetime import date

import pandas as pd
import pytest

from alphaflow.core.data_access.kdb_client import KDBClient
from alphaflow.data_dumper.extractors.bbg_extractor import BBGExtractor
from alphaflow.data_dumper.extractors.kdb_extractor import KDB_FIELDS, KDBExtractor
from alphaflow.data_dumper.runner import DataDumperRunner
from alphaflow.data_dumper.writers.csv_writer import CSVWriter

DAY = date(2024, 1, 15)
RICS = ["0700.HK", "0941.HK"]


@pytest.fixture
def extractor(system_config, fake_desktool):
    frame = pd.DataFrame({"RIC": RICS, "timestamp": [DAY, DAY], "close": [1.0, 2.0]})
    desktool = fake_desktool(kdb_result=frame)
    return KDBExtractor(KDBClient(system_config.kdb, desktool=desktool)), desktool


class TestKDBExtractor:
    def test_supports_the_six_kdb_data_types(self, extractor):
        supported = extractor[0].supported_data_types
        assert set(supported) == set(KDB_FIELDS)
        assert {"depth_open", "depth_close", "depth_intraday", "indic_open", "indic_close", "minute_bar"} <= set(supported)

    def test_an_unsupported_data_type_is_rejected(self, extractor):
        with pytest.raises(ValueError, match="does not handle 'southbound'"):
            extractor[0].extract("southbound", "HK", RICS, DAY)

    def test_the_configured_table_name_is_used(self, extractor, system_config):
        instance, desktool = extractor
        instance.extract("minute_bar", "HK", RICS, DAY)
        assert desktool.calls[0][1]["table"] == system_config.kdb.table_for("minute_bar")

    def test_a_dump_queries_a_single_day(self, extractor):
        instance, desktool = extractor
        instance.extract("minute_bar", "HK", RICS, DAY)
        assert desktool.calls[0][1]["start_date"] == DAY == desktool.calls[0][1]["end_date"]

    def test_output_is_ric_keyed(self, extractor):
        assert "RIC" in extractor[0].extract("minute_bar", "HK", RICS, DAY).columns


class TestBBGExtractorPlaceholder:
    def test_extract_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            BBGExtractor().extract("bbg", "HK", RICS, DAY)

    def test_it_still_declares_the_bbg_data_type(self):
        assert BBGExtractor().supports("bbg")


class TestCSVWriter:
    def test_writes_to_the_spec_path_with_ric_first(self, system_config, isolated_shared_drive):
        writer = CSVWriter(system_config.storage)
        writer.write(pd.DataFrame({"close": [1.0], "RIC": ["0700.HK"]}), "HK", "minute_bar", DAY)
        path = isolated_shared_drive / "raw_data" / "HK" / "minute_bar" / "20240115.csv"
        assert path.is_file() and pd.read_csv(path).columns[0] == "RIC"

    def test_a_frame_without_a_ric_column_is_rejected(self, system_config):
        with pytest.raises(ValueError, match="must carry a RIC column"):
            CSVWriter(system_config.storage).write(pd.DataFrame({"close": [1.0]}), "HK", "minute_bar", DAY)

    def test_file_exists_reflects_the_write(self, system_config):
        writer = CSVWriter(system_config.storage)
        assert writer.file_exists("HK", "minute_bar", DAY) is False
        writer.write(pd.DataFrame({"RIC": ["0700.HK"]}), "HK", "minute_bar", DAY)
        assert writer.file_exists("HK", "minute_bar", DAY) is True


class TestRunnerOrchestration:
    def test_source_none_expands_to_every_available_data_type(self, system_config):
        runner = DataDumperRunner(system_config)
        assert runner._resolve_data_types("HK", None) == system_config.available_data_types("HK")

    def test_an_unavailable_source_is_rejected(self, system_config):
        with pytest.raises(ValueError, match="not available for KR"):
            DataDumperRunner(system_config)._resolve_data_types("KR", "southbound")

    def test_date_and_range_are_mutually_exclusive(self, system_config):
        runner = DataDumperRunner(system_config)
        with pytest.raises(ValueError, match="not both"):
            runner._resolve_dates("20240115", "20240101", "20240131")
        with pytest.raises(ValueError, match="pass date=, or both"):
            runner._resolve_dates(None, "20240101", None)

    def test_a_range_expands_inclusively(self, system_config):
        assert len(DataDumperRunner(system_config)._resolve_dates(None, "20240101", "20240103")) == 3

    def test_universe_is_loaded_from_the_market_csv(self, system_config):
        assert "0700.HK" in DataDumperRunner(system_config).load_universe("HK")

    def test_existing_files_are_skipped_unless_overwrite(self, system_config, mocker):
        runner = DataDumperRunner(system_config)
        runner.writer.write(pd.DataFrame({"RIC": RICS}), "HK", "minute_bar", DAY)
        spy = mocker.spy(runner.extractors[0], "extract")
        assert runner.run(market="HK", date="20240115", source="minute_bar") == {"minute_bar": []}
        assert spy.call_count == 0

    def test_one_failing_data_type_does_not_abort_the_rest(self, system_config, mocker):
        runner = DataDumperRunner(system_config)
        frame = pd.DataFrame({"RIC": RICS, "close": [1.0, 2.0]})
        mocker.patch.object(runner.extractors[0], "extract",
                            side_effect=[ValueError("kdb down"), frame, frame, frame, frame, frame])
        written = runner.run(market="HK", date="20240115")
        assert written["depth_open"] == [] and written["minute_bar"] == ["20240115"]
