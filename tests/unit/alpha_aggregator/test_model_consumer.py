from datetime import date

import pandas as pd
import pytest

from alphaflow.alpha_aggregator.consumer.model_consumer import ModelConsumer
from alphaflow.alpha_aggregator.storage.parquet_writer import ParquetWriter
from alphaflow.core.config.model_config import ModelConfig

TODAY = date(2024, 1, 31)
MOMENTUM, IMBALANCE, RETURN = "momentum_price_21d", "order_imbalance_1430", "open_to_close_1d"


class FakeKDBWriter:
    def __init__(self, scores):
        self.scores = scores
        self.queries = []

    def query_latest_scores(self, market, alpha_ids):
        self.queries.append((market, list(alpha_ids)))
        return {a: s for a, s in self.scores.items() if a in alpha_ids}


def weighted_model(registry, weights, save_to_parquet=False):
    registry.models["hk_momentum"] = ModelConfig(
        model_id="hk_momentum", market="HK", alpha_ids=[MOMENTUM, IMBALANCE], rtn_alpha_id=RETURN,
        start_date="20230101", end_date="20231231", save_to_parquet=save_to_parquet, alpha_weights=weights)
    return registry


class TestComposite:
    def test_composite_is_the_weighted_sum(self, system_config, registry):
        registry = weighted_model(registry, {MOMENTUM: 0.6, IMBALANCE: 0.4})
        scores = {MOMENTUM: pd.Series({"0700.HK": 2.0}), IMBALANCE: pd.Series({"0700.HK": 1.0})}
        consumer = ModelConsumer(system_config, registry, FakeKDBWriter(scores))
        assert consumer.get_composite_score("hk_momentum")["0700.HK"] == pytest.approx(0.6 * 2.0 + 0.4 * 1.0)

    def test_a_ric_missing_from_one_alpha_contributes_zero_for_it(self, system_config, registry):
        registry = weighted_model(registry, {MOMENTUM: 0.6, IMBALANCE: 0.4})
        scores = {MOMENTUM: pd.Series({"0700.HK": 2.0, "0941.HK": 5.0}), IMBALANCE: pd.Series({"0700.HK": 1.0})}
        composite = ModelConsumer(system_config, registry, FakeKDBWriter(scores)).get_composite_score("hk_momentum")
        assert composite["0941.HK"] == pytest.approx(0.6 * 5.0)

    def test_the_composite_covers_the_union_of_rics(self, system_config, registry):
        registry = weighted_model(registry, {MOMENTUM: 0.5, IMBALANCE: 0.5})
        scores = {MOMENTUM: pd.Series({"0700.HK": 1.0}), IMBALANCE: pd.Series({"0941.HK": 1.0})}
        composite = ModelConsumer(system_config, registry, FakeKDBWriter(scores)).get_composite_score("hk_momentum")
        assert set(composite.index) == {"0700.HK", "0941.HK"}

    def test_partial_coverage_is_not_renormalized(self, system_config, registry):
        registry = weighted_model(registry, {MOMENTUM: 0.6, IMBALANCE: 0.4})
        scores = {MOMENTUM: pd.Series({"0700.HK": 1.0}), IMBALANCE: pd.Series({"0941.HK": 1.0})}
        composite = ModelConsumer(system_config, registry, FakeKDBWriter(scores)).get_composite_score("hk_momentum")
        # a name covered by one alpha keeps that alpha's weight, it is not scaled back up to 1.0
        assert composite["0700.HK"] == pytest.approx(0.6) and composite["0941.HK"] == pytest.approx(0.4)

    def test_the_rtn_alpha_is_not_part_of_the_composite(self, system_config, registry):
        registry = weighted_model(registry, {MOMENTUM: 0.5, IMBALANCE: 0.5})
        scores = {MOMENTUM: pd.Series({"0700.HK": 1.0}), IMBALANCE: pd.Series({"0700.HK": 1.0}),
                  RETURN: pd.Series({"0700.HK": 100.0})}
        writer = FakeKDBWriter(scores)
        composite = ModelConsumer(system_config, registry, writer).get_composite_score("hk_momentum")
        assert RETURN not in writer.queries[0][1] and composite["0700.HK"] == pytest.approx(1.0)

    def test_the_series_is_named_for_the_model_and_indexed_by_ric(self, system_config, registry):
        registry = weighted_model(registry, {MOMENTUM: 1.0, IMBALANCE: 0.0})
        scores = {MOMENTUM: pd.Series({"0700.HK": 1.0}), IMBALANCE: pd.Series({"0700.HK": 1.0})}
        composite = ModelConsumer(system_config, registry, FakeKDBWriter(scores)).get_composite_score("hk_momentum")
        assert composite.name == "hk_momentum" and composite.index.name == "RIC"


class TestGuards:
    def test_an_unoptimized_model_is_rejected(self, system_config, registry):
        with pytest.raises(ValueError, match="no alpha_weights"):
            ModelConsumer(system_config, registry, FakeKDBWriter({})).get_composite_score("hk_momentum")

    def test_no_live_scores_at_all_is_an_error(self, system_config, registry):
        registry = weighted_model(registry, {MOMENTUM: 0.6, IMBALANCE: 0.4})
        with pytest.raises(ValueError, match="no live scores available"):
            ModelConsumer(system_config, registry, FakeKDBWriter({})).get_composite_score("hk_momentum")

    def test_a_partially_available_model_still_produces_a_composite(self, system_config, registry):
        registry = weighted_model(registry, {MOMENTUM: 0.6, IMBALANCE: 0.4})
        scores = {MOMENTUM: pd.Series({"0700.HK": 2.0})}
        composite = ModelConsumer(system_config, registry, FakeKDBWriter(scores)).get_composite_score("hk_momentum")
        assert composite["0700.HK"] == pytest.approx(1.2)


class TestArchiving:
    def test_nothing_is_written_when_save_to_parquet_is_false(self, system_config, registry, isolated_shared_drive):
        registry = weighted_model(registry, {MOMENTUM: 1.0, IMBALANCE: 0.0}, save_to_parquet=False)
        scores = {MOMENTUM: pd.Series({"0700.HK": 1.0}), IMBALANCE: pd.Series({"0700.HK": 1.0})}
        ModelConsumer(system_config, registry, FakeKDBWriter(scores)).get_composite_score("hk_momentum", TODAY)
        assert not (isolated_shared_drive / "alpha_scores" / "hk_momentum").exists()

    def test_the_archive_holds_the_components_alongside_the_composite(self, system_config, registry, isolated_shared_drive):
        registry = weighted_model(registry, {MOMENTUM: 0.6, IMBALANCE: 0.4}, save_to_parquet=True)
        scores = {MOMENTUM: pd.Series({"0700.HK": 2.0}), IMBALANCE: pd.Series({"0700.HK": 1.0})}
        ModelConsumer(system_config, registry, FakeKDBWriter(scores)).get_composite_score("hk_momentum", TODAY)
        stored = pd.read_parquet(isolated_shared_drive / "alpha_scores" / "hk_momentum" / "20240131.parquet")
        assert set(stored.columns) == {"RIC", "composite", MOMENTUM, IMBALANCE}
        assert stored.set_index("RIC").loc["0700.HK", "composite"] == pytest.approx(1.6)

    def test_the_frame_puts_ric_and_composite_first(self):
        frame = ParquetWriter.build_frame({"a": pd.Series({"0700.HK": 1.0})}, pd.Series({"0700.HK": 0.5}))
        assert list(frame.columns) == ["RIC", "composite", "a"]
