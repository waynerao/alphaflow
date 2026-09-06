from datetime import date

import pandas as pd
import pytest

from alphaflow.signal_builder.storage.score_writer import ScoreWriter

DAY = date(2024, 1, 15)
MARKET, ALPHA = "HK", "momentum_price_21d"


def scores(rics=("0700.HK", "0941.HK")):
    return pd.DataFrame({"RIC": list(rics), "raw_signal": [0.1, 0.2], "score": [1.0, 2.0]})


@pytest.fixture
def writer(system_config):
    return ScoreWriter(system_config.storage)


class TestWriteAndRead:
    def test_written_to_the_spec_path(self, writer, isolated_shared_drive):
        writer.write(scores(), MARKET, ALPHA, DAY)
        assert (isolated_shared_drive / "alpha_scores" / MARKET / ALPHA / "20240115.parquet").is_file()

    def test_round_trips_the_three_columns_and_no_date(self, writer):
        writer.write(scores(), MARKET, ALPHA, DAY)
        result = writer.read(MARKET, ALPHA, DAY)
        assert list(result.columns) == ["RIC", "raw_signal", "score"]

    def test_extra_columns_are_dropped_on_write(self, writer):
        frame = scores()
        frame["scratch"] = 1
        writer.write(frame, MARKET, ALPHA, DAY)
        assert "scratch" not in writer.read(MARKET, ALPHA, DAY).columns

    def test_a_frame_missing_score_columns_is_rejected(self, writer):
        with pytest.raises(ValueError, match="missing columns"):
            writer.write(pd.DataFrame({"RIC": ["0700.HK"]}), MARKET, ALPHA, DAY)

    def test_reading_an_absent_day_reports_the_path(self, writer):
        with pytest.raises(FileNotFoundError, match="no scores at"):
            writer.read(MARKET, ALPHA, DAY)

    def test_file_exists_reflects_the_write(self, writer):
        assert writer.file_exists(MARKET, ALPHA, DAY) is False
        writer.write(scores(), MARKET, ALPHA, DAY)
        assert writer.file_exists(MARKET, ALPHA, DAY) is True


class TestRangeReads:
    def _write_days(self, writer, days):
        for day in days:
            writer.write(scores(), MARKET, ALPHA, day)

    def test_read_range_adds_a_date_column(self, writer):
        days = [date(2024, 1, d) for d in (10, 11, 12)]
        self._write_days(writer, days)
        result = writer.read_range(MARKET, ALPHA, date(2024, 1, 10), date(2024, 1, 12))
        assert list(result.columns) == ["date", "RIC", "raw_signal", "score"] and len(result) == 6

    def test_read_range_respects_the_bounds(self, writer):
        self._write_days(writer, [date(2024, 1, d) for d in (10, 11, 12)])
        assert set(writer.read_range(MARKET, ALPHA, date(2024, 1, 11), date(2024, 1, 11))["date"]) == {date(2024, 1, 11)}

    def test_an_absent_alpha_yields_an_empty_shaped_frame(self, writer):
        result = writer.read_range(MARKET, "ghost", date(2024, 1, 1), date(2024, 1, 31))
        assert result.empty and list(result.columns) == ["date", "RIC", "raw_signal", "score"]

    def test_read_dates_skips_missing_days_silently(self, writer):
        writer.write(scores(), MARKET, ALPHA, date(2024, 1, 10))
        result = writer.read_dates(MARKET, ALPHA, ["20240109", "20240110", "20240111"])
        assert set(result["date"]) == {date(2024, 1, 10)}

    def test_available_dates_lists_stored_days(self, writer):
        self._write_days(writer, [date(2024, 1, 10), date(2024, 1, 12)])
        assert writer.available_dates(MARKET, ALPHA) == ["20240110", "20240112"]
