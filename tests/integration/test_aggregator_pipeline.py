from datetime import date

import pandas as pd
import pytest

from alphaflow.alpha_aggregator.consumer.model_consumer import ModelConsumer
from alphaflow.alpha_aggregator.producer.market_producer import MarketProducerThread
from alphaflow.alpha_aggregator.runner import AlphaAggregatorRunner
from alphaflow.alpha_aggregator.storage.kdb_writer import KDBWriter
from alphaflow.core.config.model_config import ModelConfig
from alphaflow.signal_builder.runner import SignalBuilderRunner

MARKET = "HK"
TODAY = date(2024, 1, 31)
MOMENTUM, IMBALANCE, RETURN = "momentum_price_21d", "order_imbalance_1430", "open_to_close_1d"


class RecordingKDBWriter(KDBWriter):
    """Captures what the producer would persist, and serves it back to the consumer -
    standing in for the kdb+ round trip."""

    def __init__(self, output_config):
        self.config = output_config
        self.rows = pd.DataFrame()

    def write_individual_scores(self, market, daily_scores, intraday_scores):
        frame = super().write_individual_scores(market, daily_scores, intraday_scores)
        self.rows = pd.concat([self.rows, frame], ignore_index=True) if not self.rows.empty else frame
        return frame

    def _insert(self, table, frame):
        return None

    def query_latest_scores(self, market, alpha_ids):
        if self.rows.empty:
            return {}
        subset = self.rows[(self.rows["market"] == market) & (self.rows["alpha_id"].isin(alpha_ids))]
        return {alpha_id: group.sort_values("timestamp").groupby("RIC")["score"].last().astype(float)
                for alpha_id, group in subset.groupby("alpha_id")}


@pytest.fixture
def weighted_registry(registry):
    registry.models["hk_momentum"] = ModelConfig(
        model_id="hk_momentum", market=MARKET, alpha_ids=[MOMENTUM, IMBALANCE], rtn_alpha_id=RETURN,
        start_date="20230101", end_date="20231231", save_to_parquet=True,
        alpha_weights={MOMENTUM: 0.6, IMBALANCE: 0.4})
    return registry


@pytest.fixture
def wired(system_config, weighted_registry, production_config, raw_data_window, mocker, depth_frame, minute_bar_frame):
    """A producer on the live path, with PITDataManager stubbed to return fixture frames."""
    raw_data_window([TODAY])
    mocker.patch("alphaflow.core.data_access.pit_manager.PITDataManager.get_depth", return_value=depth_frame(TODAY))
    mocker.patch("alphaflow.core.data_access.pit_manager.PITDataManager.get_minute_bar",
                 side_effect=lambda rics, market, start_date: minute_bar_frame(
                     [start_date + pd.Timedelta(days=d) for d in range((TODAY - start_date).days + 9)]))
    writer = RecordingKDBWriter(production_config.output)
    builder = SignalBuilderRunner(system_config, weighted_registry)
    producer = MarketProducerThread(MARKET, builder, weighted_registry, production_config, writer)
    return producer, writer, weighted_registry


class TestAggregatorPipeline:
    """Producer computes individual scores -> Consumer combines them on demand."""

    def test_the_producer_writes_individual_scores_only(self, wired):
        producer, writer, _ = wired
        producer.run_once(TODAY)
        assert set(writer.rows["alpha_id"]) == {MOMENTUM, IMBALANCE, RETURN}
        assert "composite" not in writer.rows.columns

    def test_rows_follow_the_production_schema(self, wired):
        producer, writer, _ = wired
        producer.run_once(TODAY)
        assert list(writer.rows.columns) == ["timestamp", "market", "alpha_id", "RIC", "raw_signal", "score", "frequency"]

    def test_daily_alphas_are_computed_once_across_cycles(self, wired):
        producer, writer, _ = wired
        producer.run_once(TODAY)
        first = writer.rows[writer.rows["alpha_id"] == MOMENTUM]["timestamp"].nunique()
        producer.run_once(TODAY)
        assert writer.rows[writer.rows["alpha_id"] == MOMENTUM]["timestamp"].nunique() == first + 1

    def test_the_consumer_combines_the_producers_scores(self, wired, system_config):
        producer, writer, registry = wired
        producer.run_once(TODAY)
        composite = ModelConsumer(system_config, registry, writer).get_composite_score("hk_momentum", TODAY)
        latest = writer.query_latest_scores(MARKET, [MOMENTUM, IMBALANCE])
        expected = latest[MOMENTUM] * 0.6 + latest[IMBALANCE].reindex(latest[MOMENTUM].index).fillna(0.0) * 0.4
        assert composite.reindex(expected.index).round(9).equals(expected.round(9))

    def test_the_composite_is_archived_but_never_sent_to_kdb(self, wired, system_config, isolated_shared_drive):
        producer, writer, registry = wired
        producer.run_once(TODAY)
        ModelConsumer(system_config, registry, writer).get_composite_score("hk_momentum", TODAY)
        archive = isolated_shared_drive / "alpha_scores" / "hk_momentum" / "20240131.parquet"
        assert archive.is_file()
        assert "composite" in pd.read_parquet(archive).columns and "composite" not in set(writer.rows["alpha_id"])

    def test_the_rtn_alpha_is_produced_but_excluded_from_the_composite(self, wired, system_config, isolated_shared_drive):
        producer, writer, registry = wired
        producer.run_once(TODAY)
        ModelConsumer(system_config, registry, writer).get_composite_score("hk_momentum", TODAY)
        archived = set(pd.read_parquet(isolated_shared_drive / "alpha_scores" / "hk_momentum" / "20240131.parquet").columns)
        # the producer computes it (models need it for backtesting) but it never weights the composite
        assert RETURN in set(writer.rows["alpha_id"]) and RETURN not in archived


class TestRunnerWiring:
    def test_active_markets_come_from_the_active_models(self, system_config, production_config, weighted_registry):
        runner = AlphaAggregatorRunner(system_config, production_config, weighted_registry)
        assert runner.active_markets() == [MARKET]

    def test_the_consumer_is_built_on_demand_not_as_a_thread(self, system_config, production_config, weighted_registry):
        runner = AlphaAggregatorRunner(system_config, production_config, weighted_registry)
        consumer = runner.get_consumer()
        assert isinstance(consumer, ModelConsumer) and runner.get_consumer() is not consumer
