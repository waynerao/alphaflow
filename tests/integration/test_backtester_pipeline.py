import pandas as pd
import pytest

from alphaflow.core.utils.storage import StoragePaths
from alphaflow.signal_builder.runner import SignalBuilderRunner
from alphaflow.strategy_backtester.engine.backtest_engine import MEASURE_ORDER
from alphaflow.strategy_backtester.runner import StrategyBacktesterRunner

MARKET = "HK"
MOMENTUM, IMBALANCE, RETURN = "momentum_price_21d", "order_imbalance_1430", "open_to_close_1d"


@pytest.fixture
def scored(system_config, registry, raw_data_window, scoring_days, date_strings):
    """Run signal_builder first - the backtester only ever reads."""
    raw_data_window(scoring_days)
    runner = SignalBuilderRunner(system_config, registry)
    dates = date_strings(scoring_days)
    for alpha_id in (MOMENTUM, IMBALANCE, RETURN):
        runner.build_multi_day_alpha(alpha_id, MARKET, dates, save=True)
    return dates


class TestBacktesterPipeline:
    """alpha scores on disk -> a measures x rtn_alpha_ids result table."""

    def test_it_produces_the_spec_table_shape(self, system_config, registry, scored, research_pit, mocker):
        mocker.patch("alphaflow.strategy_backtester.runner.PITDataManager.for_research", return_value=research_pit())
        table = StrategyBacktesterRunner(system_config, registry).run(MOMENTUM, MARKET, scored, RETURN)[MOMENTUM]
        assert table.columns[0] == "measure" and RETURN in table.columns
        assert set(MEASURE_ORDER) <= set(table["measure"])

    def test_barra_rows_appear_for_every_discovered_factor(self, system_config, registry, scored, research_pit, mocker):
        mocker.patch("alphaflow.strategy_backtester.runner.PITDataManager.for_research", return_value=research_pit())
        table = StrategyBacktesterRunner(system_config, registry).run(MOMENTUM, MARKET, scored, RETURN)[MOMENTUM]
        assert sum(1 for m in table["measure"] if m.startswith("barra_")) == 6

    def test_two_return_definitions_become_two_columns(self, system_config, registry, scored, research_pit, mocker):
        mocker.patch("alphaflow.strategy_backtester.runner.PITDataManager.for_research", return_value=research_pit())
        # reuse the same rtn alpha under a second id by copying its stored scores
        paths = StoragePaths(system_config.storage)
        source = paths.alpha_score_dir(MARKET, RETURN)
        for parquet in source.glob("*.parquet"):
            target = StoragePaths.ensure_parent(paths.alpha_score(MARKET, "alt_return", parquet.stem))
            pd.read_parquet(parquet).to_parquet(target, index=False)
        registry.registry_config.alphas.append(registry.require_entry(RETURN).model_copy(update={"alpha_id": "alt_return"}))
        table = StrategyBacktesterRunner(system_config, registry).run(MOMENTUM, MARKET, scored, [RETURN, "alt_return"])[MOMENTUM]
        assert list(table.columns) == ["measure", RETURN, "alt_return"]

    def test_results_are_always_written_and_protected(self, system_config, registry, scored, research_pit,
                                                      mocker, isolated_shared_drive):
        mocker.patch("alphaflow.strategy_backtester.runner.PITDataManager.for_research", return_value=research_pit())
        runner = StrategyBacktesterRunner(system_config, registry)
        runner.run(MOMENTUM, MARKET, scored, RETURN)
        assert list((isolated_shared_drive / "backtest_results" / MARKET).glob("*.parquet"))
        with pytest.raises(FileExistsError):
            runner.run(MOMENTUM, MARKET, scored, RETURN)

    def test_a_non_rtn_return_definition_is_rejected(self, system_config, registry, scored):
        with pytest.raises(ValueError, match="must be 'rtn'"):
            StrategyBacktesterRunner(system_config, registry).run(MOMENTUM, MARKET, scored, MOMENTUM)

    def test_missing_scores_are_reported_rather_than_silently_empty(self, system_config, registry, scored):
        with pytest.raises(FileNotFoundError, match="run signal_builder first"):
            StrategyBacktesterRunner(system_config, registry).run(MOMENTUM, "CN", scored, RETURN)
