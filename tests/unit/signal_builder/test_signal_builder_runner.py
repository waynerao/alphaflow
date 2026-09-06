from datetime import date, timedelta

import pytest

from alphaflow.core.config.exceptions import ConfigValidationError
from alphaflow.core.utils.storage import StoragePaths
from alphaflow.signal_builder.runner import SOURCE_LIVE, SOURCE_RAW_DATA, SignalBuilderRunner

MARKET = "HK"
AS_OF = date(2024, 1, 31)
MOMENTUM, IMBALANCE, RETURN = "momentum_price_21d", "order_imbalance_1430", "open_to_close_1d"


@pytest.fixture
def runner(system_config, registry):
    return SignalBuilderRunner(system_config, registry)


@pytest.fixture
def seeded_raw_data(system_config, minute_bar_frame, depth_frame):
    """Write the CSVs data_dumper would have produced, over a window wide enough for the
    21-day lookback and the 1-day forward horizon."""
    paths = StoragePaths(system_config.storage)
    days = [AS_OF - timedelta(days=offset) for offset in range(35, -3, -1)]
    for day in days:
        frame = minute_bar_frame([day])
        StoragePaths.ensure_parent(paths.raw_data(MARKET, "minute_bar", day))
        frame.to_csv(paths.raw_data(MARKET, "minute_bar", day), index=False)
    StoragePaths.ensure_parent(paths.raw_data(MARKET, "depth_intraday", AS_OF))
    depth_frame(AS_OF).to_csv(paths.raw_data(MARKET, "depth_intraday", AS_OF), index=False)
    return days


class TestAlphaResolution:
    def test_all_expands_to_every_active_alpha(self, runner):
        assert set(runner._resolve_alpha_ids("all", MARKET)) == {MOMENTUM, IMBALANCE, RETURN}

    def test_a_string_and_a_list_are_both_accepted(self, runner):
        assert runner._resolve_alpha_ids(MOMENTUM, MARKET) == [MOMENTUM]
        assert runner._resolve_alpha_ids([MOMENTUM, IMBALANCE], MARKET) == [MOMENTUM, IMBALANCE]

    def test_an_unregistered_alpha_is_rejected(self, runner):
        with pytest.raises(ConfigValidationError, match="not in registry"):
            runner._resolve_alpha_ids("ghost", MARKET)


class TestBuildOneDay:
    def test_returns_the_three_score_columns_with_no_date(self, runner, seeded_raw_data):
        result = runner.build_one_day_alpha(MOMENTUM, MARKET, "20240131")
        assert list(result[MOMENTUM].columns) == ["RIC", "raw_signal", "score"]

    def test_raw_signal_and_score_differ_when_normalization_scales(self, runner, seeded_raw_data):
        frame = runner.build_one_day_alpha(MOMENTUM, MARKET, "20240131")[MOMENTUM]
        assert not frame["raw_signal"].equals(frame["score"])

    def test_save_false_writes_nothing(self, runner, seeded_raw_data):
        runner.build_one_day_alpha(MOMENTUM, MARKET, "20240131", save=False)
        assert runner.score_writer.file_exists(MARKET, MOMENTUM, AS_OF) is False

    def test_save_true_persists_the_scores(self, runner, seeded_raw_data):
        runner.build_one_day_alpha(MOMENTUM, MARKET, "20240131", save=True)
        assert runner.score_writer.file_exists(MARKET, MOMENTUM, AS_OF) is True

    def test_existing_scores_are_not_recomputed_unless_overwrite(self, runner, seeded_raw_data, mocker):
        runner.build_one_day_alpha(MOMENTUM, MARKET, "20240131", save=True)
        spy = mocker.spy(runner, "_load_and_join")
        runner.build_one_day_alpha(MOMENTUM, MARKET, "20240131", save=True, overwrite=False)
        assert spy.call_count == 0
        runner.build_one_day_alpha(MOMENTUM, MARKET, "20240131", save=True, overwrite=True)
        assert spy.call_count == 1

    def test_an_alpha_outside_the_market_scope_is_skipped(self, runner, seeded_raw_data):
        # order_imbalance_1430 is scoped to HK and CN only
        assert runner.build_one_day_alpha(IMBALANCE, "JP", "20240131") == {}

    def test_several_alphas_are_returned_keyed_by_id(self, runner, seeded_raw_data):
        result = runner.build_one_day_alpha([MOMENTUM, IMBALANCE], MARKET, "20240131")
        assert set(result) == {MOMENTUM, IMBALANCE}


class TestImplicitUniverse:
    def test_only_rics_in_the_universe_csv_survive(self, runner, system_config, minute_bar_frame):
        paths = StoragePaths(system_config.storage)
        days = [AS_OF - timedelta(days=offset) for offset in range(35, -1, -1)]
        for day in days:
            frame = minute_bar_frame([day], rics=["0700.HK", "NOT_IN_UNIVERSE.HK"])
            StoragePaths.ensure_parent(paths.raw_data(MARKET, "minute_bar", day))
            frame.to_csv(paths.raw_data(MARKET, "minute_bar", day), index=False)
        result = runner.build_one_day_alpha(MOMENTUM, MARKET, "20240131")[MOMENTUM]
        assert set(result["RIC"]) == {"0700.HK"}

    def test_rows_with_nulls_are_dropped(self, runner, system_config, minute_bar_frame):
        paths = StoragePaths(system_config.storage)
        days = [AS_OF - timedelta(days=offset) for offset in range(35, -1, -1)]
        for day in days:
            frame = minute_bar_frame([day])
            frame.loc[frame["RIC"] == "0941.HK", "close"] = None
            StoragePaths.ensure_parent(paths.raw_data(MARKET, "minute_bar", day))
            frame.to_csv(paths.raw_data(MARKET, "minute_bar", day), index=False)
        result = runner.build_one_day_alpha(MOMENTUM, MARKET, "20240131")[MOMENTUM]
        assert "0941.HK" not in set(result["RIC"])

    def test_no_surviving_rics_is_an_explicit_error(self, runner, system_config, minute_bar_frame):
        paths = StoragePaths(system_config.storage)
        for day in [AS_OF - timedelta(days=offset) for offset in range(35, -1, -1)]:
            frame = minute_bar_frame([day], rics=["GHOST.HK"])
            StoragePaths.ensure_parent(paths.raw_data(MARKET, "minute_bar", day))
            frame.to_csv(paths.raw_data(MARKET, "minute_bar", day), index=False)
        with pytest.raises(ValueError, match="no RICs survive the join"):
            runner.build_one_day_alpha(MOMENTUM, MARKET, "20240131")

    def test_missing_raw_data_reports_the_data_type(self, runner):
        with pytest.raises(FileNotFoundError, match="no minute_bar CSV"):
            runner.build_one_day_alpha(MOMENTUM, MARKET, "20240131")


class TestWorkingDataInjection:
    def test_working_data_is_set_before_compute_runs(self, runner, seeded_raw_data, mocker):
        captured = {}
        original = runner._load_and_join

        def spy(alpha, market, as_of, source):
            frame = original(alpha, market, as_of, source)
            captured["columns"] = list(frame.columns)
            return frame

        mocker.patch.object(runner, "_load_and_join", side_effect=spy)
        runner.build_one_day_alpha(MOMENTUM, MARKET, "20240131")
        assert "RIC" in captured["columns"] and "close" in captured["columns"]


class TestHistoryWindow:
    def test_low_alphas_reach_backwards(self, runner, seeded_raw_data):
        alpha = runner._build_alpha(MOMENTUM, AS_OF, SOURCE_RAW_DATA)
        days = SignalBuilderRunner._history_days(alpha, AS_OF)
        assert days[-1] == AS_OF and days[0] < AS_OF and len(days) == 21 + 1 + 7 + 1

    def test_rtn_alphas_reach_forwards(self, runner, seeded_raw_data):
        alpha = runner._build_alpha(RETURN, AS_OF, SOURCE_RAW_DATA)
        days = SignalBuilderRunner._history_days(alpha, AS_OF)
        assert days[0] == AS_OF and days[-1] > AS_OF

    def test_intraday_alphas_load_only_the_day(self, runner, seeded_raw_data):
        alpha = runner._build_alpha(IMBALANCE, AS_OF, SOURCE_RAW_DATA)
        assert SignalBuilderRunner._history_days(alpha, AS_OF) == [AS_OF]

    def test_a_rtn_alphas_pit_cut_is_extended_past_its_horizon(self, runner, seeded_raw_data):
        alpha = runner._build_alpha(RETURN, AS_OF, SOURCE_RAW_DATA)
        assert alpha.data.as_of_date > AS_OF


class TestSourceModes:
    def test_live_mode_reads_the_pit_manager_not_the_csvs(self, runner, mocker, minute_bar_frame):
        frame = minute_bar_frame([AS_OF])
        mocker.patch("alphaflow.core.data_access.pit_manager.PITDataManager.get_depth", return_value=frame)
        alpha = runner._build_alpha(IMBALANCE, AS_OF, SOURCE_LIVE)
        assert alpha.data.mode == "production"

    def test_live_mode_dispatches_to_the_right_getter(self, runner, mocker, depth_frame):
        expected = depth_frame(AS_OF)
        patched = mocker.patch("alphaflow.core.data_access.pit_manager.PITDataManager.get_depth", return_value=expected)
        result = runner.build_one_day_alpha(IMBALANCE, MARKET, "20240131", source="live")
        assert patched.call_args.args[3] == "intraday" and not result[IMBALANCE].empty

    def test_an_unknown_source_is_rejected(self, runner, seeded_raw_data):
        with pytest.raises(ValueError, match="source must be"):
            runner.build_one_day_alpha(MOMENTUM, MARKET, "20240131", source="telepathy")


class TestBuildMultiDay:
    def test_result_carries_a_date_column_and_stacks_days(self, runner, seeded_raw_data):
        dates = ["20240129", "20240130", "20240131"]
        result = runner.build_multi_day_alpha(MOMENTUM, MARKET, dates)[MOMENTUM]
        assert list(result.columns) == ["date", "RIC", "raw_signal", "score"]
        assert result["date"].nunique() == 3

    def test_an_empty_date_list_is_rejected(self, runner):
        with pytest.raises(ValueError, match="dates must not be empty"):
            runner.build_multi_day_alpha(MOMENTUM, MARKET, [])
