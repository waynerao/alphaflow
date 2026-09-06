from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from alphaflow.strategy_backtester.engine.backtest_engine import MEASURE_ORDER, BacktestEngine
from alphaflow.strategy_backtester.regression.cross_sectional_ols import CrossSectionalOLS
from alphaflow.strategy_backtester.reporting.report_generator import ReportGenerator
from alphaflow.strategy_backtester.risk.barra_exposure import BarraExposure, discover_factors

DAYS = [date(2024, 1, 1) + timedelta(days=d) for d in range(10)]


class TestPooledOLS:
    def test_recovers_a_planted_coefficient(self, correlated_frames):
        alpha, returns = correlated_frames(beta=0.5, noise=0.1, seed=5)
        assert CrossSectionalOLS().fit(alpha, returns)["coefficient"] == pytest.approx(0.5, abs=0.05)

    def test_matches_statsmodels_exactly(self, correlated_frames):
        alpha, returns = correlated_frames(beta=0.3, noise=0.4, seed=6)
        result = CrossSectionalOLS().fit(alpha, returns)
        merged = alpha.merge(returns[["date", "RIC", "score"]].rename(columns={"score": "fr"}), on=["date", "RIC"])
        truth = sm.OLS(merged["fr"], sm.add_constant(merged["score"])).fit()
        assert result["coefficient"] == pytest.approx(truth.params.iloc[1])
        assert result["r_squared"] == pytest.approx(truth.rsquared)
        assert result["t_stat"] == pytest.approx(truth.tvalues.iloc[1])
        assert result["intercept"] == pytest.approx(truth.params.iloc[0])

    def test_it_pools_across_dates_rather_than_averaging_per_day(self, correlated_frames):
        alpha, returns = correlated_frames(beta=0.5, noise=0.2, seed=7)
        merged = alpha.merge(returns[["date", "RIC"]], on=["date", "RIC"])
        assert CrossSectionalOLS().fit(alpha, returns)["n_stocks"] == merged["RIC"].nunique()

    def test_a_perfect_fit_has_r_squared_one(self):
        rows_a = [{"date": DAYS[0], "RIC": f"{i}.HK", "score": float(i)} for i in range(20)]
        rows_r = [{"date": DAYS[0], "RIC": f"{i}.HK", "score": 2.0 * i + 1.0} for i in range(20)]
        result = CrossSectionalOLS().fit(pd.DataFrame(rows_a), pd.DataFrame(rows_r))
        assert result["r_squared"] == pytest.approx(1.0) and result["coefficient"] == pytest.approx(2.0)
        assert result["intercept"] == pytest.approx(1.0)

    def test_too_few_observations_yield_nans(self):
        alpha = pd.DataFrame([{"date": DAYS[0], "RIC": "A.HK", "score": 1.0}])
        assert all(np.isnan(v) for v in CrossSectionalOLS().fit(alpha, alpha).values())

    def test_a_constant_score_yields_nans(self):
        rows = [{"date": DAYS[0], "RIC": f"{i}.HK", "score": 1.0} for i in range(10)]
        alpha = pd.DataFrame(rows)
        returns = pd.DataFrame([{**r, "score": float(i)} for i, r in enumerate(rows)])
        assert np.isnan(CrossSectionalOLS().fit(alpha, returns)["coefficient"])


class TestBarraExposure:
    def test_factors_are_discovered_not_hardcoded(self, risk_frame_factory):
        frame = risk_frame_factory(factors=["something", "entirely_new", "ind_x"])
        assert set(discover_factors(frame)) == {"something", "entirely_new", "ind_x"}

    def test_identifier_columns_are_never_treated_as_factors(self, risk_frame_factory):
        frame = risk_frame_factory()
        frame["date"] = DAYS[0]
        assert not {"RIC", "model", "date"} & set(discover_factors(frame))

    def test_exposure_is_the_score_weighted_mean(self, research_pit):
        risk = pd.DataFrame({"RIC": ["A.HK", "B.HK"], "model": ["ASE2S"] * 2, "size": [1.0, 3.0]})
        scores = pd.DataFrame({"date": [DAYS[0]] * 2, "RIC": ["A.HK", "B.HK"], "score": [1.0, 3.0]})
        result = BarraExposure(research_pit(risk=risk)).compute(scores, "HK", [DAYS[0]])
        # weights 0.25 / 0.75 -> 0.25*1 + 0.75*3 = 2.5
        assert result["barra_size"] == pytest.approx(2.5)

    def test_absolute_scores_are_used_so_a_short_leg_does_not_cancel(self, research_pit):
        risk = pd.DataFrame({"RIC": ["A.HK", "B.HK"], "model": ["ASE2S"] * 2, "size": [1.0, 1.0]})
        scores = pd.DataFrame({"date": [DAYS[0]] * 2, "RIC": ["A.HK", "B.HK"], "score": [1.0, -1.0]})
        assert BarraExposure(research_pit(risk=risk)).compute(scores, "HK", [DAYS[0]])["barra_size"] == pytest.approx(1.0)

    def test_unavailable_risk_data_degrades_to_no_exposures(self, system_config, fake_desktool):
        from alphaflow.core.data_access.pit_manager import PITDataManager
        manager = PITDataManager.for_production(system_config, desktool=fake_desktool(kdb_result=pd.DataFrame()))
        assert BarraExposure(manager).compute(pd.DataFrame({"RIC": ["A.HK"], "score": [1.0]}), "HK", [DAYS[0]]) == {}


class TestReportShape:
    def test_rows_are_measures_and_columns_are_rtn_alpha_ids(self, system_config):
        reporter = ReportGenerator(system_config.storage)
        table = reporter.assemble({"rtn_a": {"mean_IC": 0.04, "sharpe": 1.2},
                                   "rtn_b": {"mean_IC": 0.03, "sharpe": 1.0}})
        assert list(table.columns) == ["measure", "rtn_a", "rtn_b"]
        assert set(table["measure"]) == {"mean_IC", "sharpe"}

    def test_measures_appear_in_the_canonical_order(self, system_config):
        reporter = ReportGenerator(system_config.storage)
        scrambled = {m: 1.0 for m in reversed(MEASURE_ORDER)}
        table = reporter.assemble({"rtn_a": scrambled})
        assert list(table["measure"]) == MEASURE_ORDER

    def test_barra_rows_follow_the_fixed_measures_and_are_sorted(self, system_config):
        reporter = ReportGenerator(system_config.storage)
        table = reporter.assemble({"rtn_a": {"sharpe": 1.0, "barra_size": 0.1, "barra_mkt": 0.9}})
        assert list(table["measure"]) == ["sharpe", "barra_mkt", "barra_size"]

    def test_results_are_always_saved_and_refuse_to_clobber(self, system_config, isolated_shared_drive):
        reporter = ReportGenerator(system_config.storage)
        table = reporter.assemble({"rtn_a": {"sharpe": 1.0}})
        path = reporter.save(table, "HK", "alpha_x", "20230101", "20231231")
        assert (isolated_shared_drive / "backtest_results" / "HK" / "alpha_x_20230101-20231231.parquet").is_file()
        with pytest.raises(FileExistsError, match="pass overwrite=True"):
            reporter.save(table, "HK", "alpha_x", "20230101", "20231231")
        assert reporter.save(table, "HK", "alpha_x", "20230101", "20231231", overwrite=True) == path


class TestBacktestEngine:
    def test_it_produces_every_fixed_measure(self, correlated_frames, research_pit):
        alpha, returns = correlated_frames(beta=0.5, noise=0.3, seed=8)
        result = BacktestEngine(research_pit()).run(alpha, returns, "HK", DAYS)
        assert set(MEASURE_ORDER) <= set(result)

    def test_barra_measures_are_included_when_risk_data_is_available(self, correlated_frames, research_pit):
        alpha, returns = correlated_frames(beta=0.5, noise=0.3, seed=9)
        result = BacktestEngine(research_pit()).run(alpha, returns, "HK", DAYS)
        assert any(k.startswith("barra_") for k in result)

    def test_it_runs_without_a_data_manager(self, correlated_frames):
        alpha, returns = correlated_frames(beta=0.5, noise=0.3, seed=10)
        result = BacktestEngine().run(alpha, returns, "HK", DAYS)
        assert not any(k.startswith("barra_") for k in result) and "mean_IC" in result
