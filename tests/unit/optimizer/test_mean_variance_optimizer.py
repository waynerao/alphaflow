import shutil
import tomllib

import numpy as np
import pandas as pd
import pytest

from alphaflow.core.config.exceptions import ConfigValidationError
from alphaflow.core.config.optimizer_config import MeanVarianceConfig
from alphaflow.optimizer.optimizers.mean_variance_optimizer import MeanVarianceOptimizer
from alphaflow.optimizer.risk.barra_decomposer import BarraDecomposer
from alphaflow.optimizer.storage.model_config_writer import ModelConfigWriter
from alphaflow.optimizer.storage.result_writer import PORTFOLIO_ROW, ResultWriter


@pytest.fixture
def returns():
    rng = np.random.default_rng(11)
    n = 250
    strong = rng.normal(0.0010, 0.010, n)
    weak = rng.normal(0.0002, 0.010, n)
    return pd.DataFrame({"strong": strong, "weak": weak, "twin": strong + rng.normal(0, 0.0005, n)})


def config(**overrides):
    return MeanVarianceConfig(**{"max_single_alpha_weight": 0.8, "min_single_alpha_weight": 0.0, "risk_aversion": 1.0, **overrides})


class TestConstraints:
    def test_weights_sum_to_one(self, returns):
        assert MeanVarianceOptimizer(config()).optimize(returns).sum() == pytest.approx(1.0)

    def test_weights_respect_the_box(self, returns):
        weights = MeanVarianceOptimizer(config(min_single_alpha_weight=0.2, max_single_alpha_weight=0.4)).optimize(returns)
        assert (weights >= 0.2 - 1e-6).all() and (weights <= 0.4 + 1e-6).all()

    def test_infeasible_bounds_are_rejected_not_silently_renormalized(self, returns):
        with pytest.raises(ValueError, match="cannot sum to 1 across 3"):
            MeanVarianceOptimizer(config(max_single_alpha_weight=0.2)).optimize(returns)

    def test_a_min_that_overshoots_is_rejected(self, returns):
        with pytest.raises(ValueError, match="cannot sum to 1"):
            MeanVarianceOptimizer(config(min_single_alpha_weight=0.5, max_single_alpha_weight=0.9)).optimize(returns)

    def test_a_single_alpha_takes_the_whole_book_despite_the_cap(self):
        # The 0.8 cap is a diversification constraint; with one alpha, sum(w)=1 must win
        weights = MeanVarianceOptimizer(config()).optimize(pd.DataFrame({"only": [0.01, 0.02, 0.03]}))
        assert weights.to_dict() == {"only": 1.0}

    def test_no_columns_is_rejected(self):
        with pytest.raises(ValueError, match="no alpha columns"):
            MeanVarianceOptimizer(config()).optimize(pd.DataFrame())

    def test_the_index_is_the_alpha_ids(self, returns):
        assert list(MeanVarianceOptimizer(config()).optimize(returns).index) == list(returns.columns)


class TestObjective:
    def test_the_better_alpha_is_favoured_over_the_weaker(self, returns):
        weights = MeanVarianceOptimizer(config()).optimize(returns)
        assert weights["strong"] + weights["twin"] > weights["weak"]

    def test_higher_risk_aversion_moves_away_from_the_riskiest_alpha(self):
        rng = np.random.default_rng(12)
        frame = pd.DataFrame({"calm": rng.normal(0.001, 0.005, 300), "wild": rng.normal(0.0012, 0.05, 300)})
        timid = MeanVarianceOptimizer(config(risk_aversion=50.0)).optimize(frame)
        bold = MeanVarianceOptimizer(config(risk_aversion=0.01)).optimize(frame)
        assert timid["calm"] > bold["calm"]

    def test_a_dominated_alpha_is_driven_to_the_lower_bound(self):
        rng = np.random.default_rng(13)
        good = rng.normal(0.002, 0.01, 300)
        frame = pd.DataFrame({"good": good, "bad": rng.normal(-0.002, 0.02, 300)})
        weights = MeanVarianceOptimizer(config(max_single_alpha_weight=1.0)).optimize(frame)
        assert weights["bad"] == pytest.approx(0.0, abs=1e-6)

    def test_the_per_alpha_cap_still_forces_weight_onto_a_dominated_alpha(self):
        rng = np.random.default_rng(13)
        good = rng.normal(0.002, 0.01, 300)
        frame = pd.DataFrame({"good": good, "bad": rng.normal(-0.002, 0.02, 300)})
        # max 0.8 across two alphas leaves 0.2 that has to go somewhere
        weights = MeanVarianceOptimizer(config(max_single_alpha_weight=0.8)).optimize(frame)
        assert weights["good"] == pytest.approx(0.8) and weights["bad"] == pytest.approx(0.2)


class TestPortfolioStats:
    def test_stats_are_computed_on_the_weighted_series(self, returns):
        weights = pd.Series({"strong": 1.0, "weak": 0.0, "twin": 0.0})
        stats = MeanVarianceOptimizer(config()).compute_portfolio_stats(returns, weights)
        assert stats["standalone_mean_return"] == pytest.approx(returns["strong"].mean())

    def test_a_missing_alpha_in_the_weights_contributes_zero(self, returns):
        stats = MeanVarianceOptimizer(config()).compute_portfolio_stats(returns, pd.Series({"strong": 1.0}))
        assert stats["standalone_mean_return"] == pytest.approx(returns["strong"].mean())


class TestBarraDecomposer:
    def test_combined_exposure_is_the_weighted_sum(self):
        standalone = {"a": {"barra_mkt": 1.0}, "b": {"barra_mkt": 0.0}}
        weights = pd.Series({"a": 0.25, "b": 0.75})
        assert BarraDecomposer().compute_combined(standalone, weights)["barra_mkt"] == pytest.approx(0.25)

    def test_a_factor_absent_for_one_alpha_contributes_zero_rather_than_nan(self):
        standalone = {"a": {"barra_mkt": 1.0, "barra_size": 0.4}, "b": {"barra_mkt": 1.0}}
        result = BarraDecomposer().compute_combined(standalone, pd.Series({"a": 0.5, "b": 0.5}))
        assert result["barra_size"] == pytest.approx(0.2) and result["barra_mkt"] == pytest.approx(1.0)

    def test_no_exposures_yields_nothing(self):
        assert BarraDecomposer().compute_combined({}, pd.Series({"a": 1.0})) == {}

    def test_standalone_without_a_data_manager_is_empty(self):
        assert BarraDecomposer().compute_standalone(pd.DataFrame(), "HK", []) == {}


class TestResultWriter:
    @pytest.fixture
    def table(self, system_config):
        writer = ResultWriter(system_config.storage)
        return writer, writer.build_table(
            pd.Series({"a": 0.6, "b": 0.4}),
            {"a": {"standalone_sharpe": 1.1, "standalone_std": 0.015, "standalone_mean_return": 0.001},
             "b": {"standalone_sharpe": 0.85, "standalone_std": 0.018, "standalone_mean_return": 0.0008}},
            {"a": 0.35, "b": 0.35},
            {"a": {"barra_mkt": 0.92}, "b": {"barra_mkt": 0.88}},
            {"standalone_sharpe": 1.35, "standalone_std": 0.012, "standalone_mean_return": 0.0011},
            {"barra_mkt": 0.90})

    def test_one_row_per_alpha_plus_a_portfolio_row(self, table):
        _, result = table
        assert list(result["row_id"]) == ["a", "b", PORTFOLIO_ROW]

    def test_portfolio_weight_is_the_total(self, table):
        _, result = table
        assert result.set_index("row_id").loc[PORTFOLIO_ROW, "weight"] == pytest.approx(1.0)

    def test_portfolio_has_no_average_correlation(self, table):
        _, result = table
        assert np.isnan(result.set_index("row_id").loc[PORTFOLIO_ROW, "avg_corr"])

    def test_leading_columns_come_first_then_barra(self, table):
        _, result = table
        assert list(result.columns) == ["row_id", "weight", "sharpe", "std", "mean_return", "avg_corr", "barra_mkt"]

    def test_saving_refuses_to_clobber_without_overwrite(self, table, isolated_shared_drive):
        writer, result = table
        writer.save(result, "HK", "m1")
        assert (isolated_shared_drive / "optimizer_results" / "HK" / "m1.parquet").is_file()
        with pytest.raises(FileExistsError):
            writer.save(result, "HK", "m1")
        writer.save(result, "HK", "m1", overwrite=True)


class TestModelConfigWriter:
    @pytest.fixture
    def models_dir(self, tmp_path):
        shutil.copy("configs/models/hk_momentum.toml", tmp_path / "hk_momentum.toml")
        return tmp_path

    def test_only_the_weights_section_changes(self, models_dir):
        before = (models_dir / "hk_momentum.toml").read_text()
        ModelConfigWriter().write_weights("hk_momentum", str(models_dir),
                                          pd.Series({"momentum_price_21d": 0.62, "order_imbalance_1430": 0.38}))
        after = (models_dir / "hk_momentum.toml").read_text()
        assert before.split("[alpha_weights]")[0] == after.split("[alpha_weights]")[0]

    def test_the_result_parses_and_keeps_every_other_key(self, models_dir):
        ModelConfigWriter().write_weights("hk_momentum", str(models_dir),
                                          pd.Series({"momentum_price_21d": 0.62, "order_imbalance_1430": 0.38}))
        parsed = tomllib.loads((models_dir / "hk_momentum.toml").read_text())
        assert parsed["alpha_weights"] == {"momentum_price_21d": 0.62, "order_imbalance_1430": 0.38}
        assert parsed["model_id"] == "hk_momentum" and parsed["rtn_alpha_id"] == "open_to_close_1d"
        assert parsed["save_to_parquet"] is True

    def test_rewriting_replaces_rather_than_appends(self, models_dir):
        writer = ModelConfigWriter()
        weights = pd.Series({"momentum_price_21d": 0.62, "order_imbalance_1430": 0.38})
        writer.write_weights("hk_momentum", str(models_dir), weights)
        writer.write_weights("hk_momentum", str(models_dir), pd.Series({"momentum_price_21d": 0.5, "order_imbalance_1430": 0.5}))
        content = (models_dir / "hk_momentum.toml").read_text()
        assert content.count("[alpha_weights]") == 1
        assert tomllib.loads(content)["alpha_weights"]["momentum_price_21d"] == 0.5

    def test_a_file_without_the_section_gains_one(self, tmp_path):
        path = tmp_path / "m2.toml"
        path.write_text('model_id = "m2"\nmarket = "HK"\n')
        ModelConfigWriter().write_weights("m2", str(tmp_path), pd.Series({"a": 1.0}))
        assert tomllib.loads(path.read_text())["alpha_weights"] == {"a": 1.0}

    def test_a_missing_model_file_is_reported(self, tmp_path):
        with pytest.raises(ConfigValidationError, match="model config not found"):
            ModelConfigWriter().write_weights("ghost", str(tmp_path), pd.Series({"a": 1.0}))

    def test_empty_weights_are_rejected(self, models_dir):
        with pytest.raises(ValueError, match="no weights to write"):
            ModelConfigWriter().write_weights("hk_momentum", str(models_dir), pd.Series(dtype=float))
