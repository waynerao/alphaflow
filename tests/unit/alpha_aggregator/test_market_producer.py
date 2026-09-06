from datetime import date

import pandas as pd
import pytest

from alphaflow.alpha_aggregator.alerting.alert import send_critical_alert
from alphaflow.alpha_aggregator.producer.daily_cache import DailyAlphaCache
from alphaflow.alpha_aggregator.producer.market_producer import MarketProducerThread, resolve_market_alpha_ids
from alphaflow.alpha_aggregator.storage.kdb_writer import FREQUENCY_DAILY, FREQUENCY_INTRADAY, KDBWriter
from alphaflow.core.config.model_config import ModelConfig
from alphaflow.core.data_access.kdb_client import KDBClient

TODAY = date(2024, 1, 31)
MOMENTUM, IMBALANCE, RETURN = "momentum_price_21d", "order_imbalance_1430", "open_to_close_1d"


class FakeSignalBuilder:
    """Stands in for SignalBuilderRunner - records what the producer asked for."""

    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def build_one_day_alpha(self, alpha_id, market, date, save=False, overwrite=False, source="raw_data"):
        self.calls.append({"alpha_id": list(alpha_id), "market": market, "date": date, "save": save, "source": source})
        if self.failure is not None:
            raise self.failure
        return {a: pd.DataFrame({"RIC": ["0700.HK", "0941.HK"], "raw_signal": [0.1, 0.2], "score": [1.0, 2.0]})
                for a in alpha_id}


@pytest.fixture
def producer(registry, production_config):
    builder = FakeSignalBuilder()
    return MarketProducerThread("HK", builder, registry, production_config), builder


class TestAlphaResolution:
    def test_the_union_includes_the_rtn_alpha(self, registry, production_config):
        assert set(resolve_market_alpha_ids("HK", production_config, registry)) == {MOMENTUM, IMBALANCE, RETURN}

    def test_an_alpha_shared_by_two_models_appears_once(self, registry, production_config):
        registry.models["second"] = ModelConfig(model_id="second", market="HK", alpha_ids=[MOMENTUM],
                                                rtn_alpha_id=RETURN, start_date="20240101", end_date="20241231")
        production_config.models.active_model_ids = ["hk_momentum", "second"]
        resolved = resolve_market_alpha_ids("HK", production_config, registry)
        assert resolved.count(MOMENTUM) == 1 and len(resolved) == len(set(resolved))

    def test_other_markets_models_are_excluded(self, registry, production_config):
        assert resolve_market_alpha_ids("JP", production_config, registry) == []

    def test_daily_and_intraday_alphas_are_split_by_type(self, producer):
        thread, _ = producer
        assert set(thread.daily_alpha_ids) == {MOMENTUM, RETURN}
        assert thread.intraday_alpha_ids == [IMBALANCE]


class TestDailyCache:
    def test_a_cache_is_stale_before_anything_is_stored(self):
        assert DailyAlphaCache().is_fresh(TODAY) is False

    def test_it_is_fresh_after_a_store_for_the_same_day(self):
        cache = DailyAlphaCache()
        cache.set(MOMENTUM, pd.Series({"0700.HK": 1.0}), TODAY)
        assert cache.is_fresh(TODAY) and cache.get(MOMENTUM) is not None

    def test_a_new_day_invalidates_the_previous_day(self):
        cache = DailyAlphaCache()
        cache.set(MOMENTUM, pd.Series({"0700.HK": 1.0}), date(2024, 1, 30))
        cache.set(RETURN, pd.Series({"0700.HK": 2.0}), TODAY)
        assert cache.get(MOMENTUM) is None and cache.is_fresh(TODAY)

    def test_force_refresh_empties_it(self):
        cache = DailyAlphaCache()
        cache.set(MOMENTUM, pd.Series({"0700.HK": 1.0}), TODAY)
        cache.force_refresh()
        assert cache.is_fresh(TODAY) is False

    def test_an_unknown_alpha_returns_none(self):
        assert DailyAlphaCache().get("ghost") is None


class TestProductionCycle:
    def test_daily_alphas_are_computed_once_then_served_from_cache(self, producer):
        thread, builder = producer
        thread.run_once(TODAY)
        thread.run_once(TODAY)
        daily_calls = [c for c in builder.calls if MOMENTUM in c["alpha_id"]]
        assert len(daily_calls) == 1

    def test_intraday_alphas_are_recomputed_every_cycle(self, producer):
        thread, builder = producer
        thread.run_once(TODAY)
        thread.run_once(TODAY)
        intraday_calls = [c for c in builder.calls if c["alpha_id"] == [IMBALANCE]]
        assert len(intraday_calls) == 2

    def test_production_reads_live_data_and_never_saves(self, producer):
        thread, builder = producer
        thread.run_once(TODAY)
        assert all(c["source"] == "live" and c["save"] is False for c in builder.calls)

    def test_a_forced_refresh_recomputes_the_daily_alphas(self, producer):
        thread, builder = producer
        thread.run_once(TODAY)
        thread.force_refresh_daily()
        thread.run_once(TODAY)
        assert len([c for c in builder.calls if MOMENTUM in c["alpha_id"]]) == 2

    def test_a_cycle_returns_every_alpha_as_a_ric_indexed_series(self, producer):
        thread, _ = producer
        result = thread.run_once(TODAY)
        assert set(result) == {MOMENTUM, IMBALANCE, RETURN}
        assert list(result[MOMENTUM].index) == ["0700.HK", "0941.HK"]


class TestFailureIsolation:
    def test_repeated_failures_stop_the_thread_and_alert(self, registry, production_config, mocker):
        alert = mocker.patch("alphaflow.alpha_aggregator.producer.market_producer.send_critical_alert")
        builder = FakeSignalBuilder(failure=RuntimeError("kdb down"))
        thread = MarketProducerThread("HK", builder, registry, production_config)
        production_config.scheduler.intraday_update_interval_sec = 1
        thread.run()
        assert thread.stopped and alert.call_count == 1
        assert "HK" in alert.call_args.args[0] and "kdb down" in alert.call_args.args[1]

    def test_the_failure_budget_is_max_restart_attempts(self, registry, production_config):
        builder = FakeSignalBuilder(failure=RuntimeError("boom"))
        thread = MarketProducerThread("HK", builder, registry, production_config)
        production_config.scheduler.intraday_update_interval_sec = 1
        thread.run()
        assert thread.failure_count == production_config.threading.max_restart_attempts

    def test_one_market_failing_leaves_another_untouched(self, registry, production_config):
        failing = MarketProducerThread("HK", FakeSignalBuilder(failure=RuntimeError("x")), registry, production_config)
        healthy = MarketProducerThread("HK", FakeSignalBuilder(), registry, production_config)
        production_config.scheduler.intraday_update_interval_sec = 1
        failing.run()
        assert failing.stopped and not healthy.stopped and healthy.run_once(TODAY)

    def test_an_alert_still_logs_critical_without_an_external_hook(self, caplog):
        with caplog.at_level("CRITICAL"):
            send_critical_alert("subject", "message")
        assert "subject: message" in caplog.text


class TestKDBWriter:
    @pytest.fixture
    def writer(self, production_config, system_config, fake_desktool):
        return KDBWriter(production_config.output, KDBClient(system_config.kdb, desktool=fake_desktool()))

    def test_rows_follow_the_production_schema(self, writer):
        rows = writer.write_individual_scores("HK", {MOMENTUM: pd.Series({"0700.HK": 1.0})}, {})
        assert list(rows.columns) == ["timestamp", "market", "alpha_id", "RIC", "raw_signal", "score", "frequency"]

    def test_daily_and_intraday_rows_are_tagged(self, writer):
        rows = writer.write_individual_scores("HK", {MOMENTUM: pd.Series({"0700.HK": 1.0})},
                                              {IMBALANCE: pd.Series({"0700.HK": 0.5})})
        tags = rows.set_index("alpha_id")["frequency"].to_dict()
        assert tags == {MOMENTUM: FREQUENCY_DAILY, IMBALANCE: FREQUENCY_INTRADAY}

    def test_no_composite_column_is_ever_written(self, writer):
        rows = writer.write_individual_scores("HK", {MOMENTUM: pd.Series({"0700.HK": 1.0})}, {})
        assert "composite" not in rows.columns and rows["alpha_id"].nunique() == 1

    def test_empty_scores_produce_no_rows(self, writer):
        assert writer.write_individual_scores("HK", {}, {}).empty
        assert writer.write_individual_scores("HK", {MOMENTUM: pd.Series(dtype=float)}, {}).empty
