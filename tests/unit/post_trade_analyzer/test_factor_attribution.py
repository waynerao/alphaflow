from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from alphaflow.post_trade_analyzer.attribution.factor_attribution import FactorAttribution
from alphaflow.post_trade_analyzer.reporting.report_generator import ReportGenerator

DAYS = [date(2024, 1, 1) + timedelta(days=d) for d in range(20)]
RICS = [f"{i:04d}.HK" for i in range(30)]


def frame(per_day):
    rows = []
    for day, values in per_day.items():
        for ric, value in values.items():
            rows.append({"date": day, "RIC": ric, "raw_signal": value, "score": value})
    return pd.DataFrame(rows)


class TestDifference:
    def test_difference_is_realized_minus_predicted(self):
        predicted = frame({DAYS[0]: {"A.HK": 0.01}})
        realized = frame({DAYS[0]: {"A.HK": 0.03}})
        result = FactorAttribution().compute_difference(predicted, realized)
        assert result["difference"].iloc[0] == pytest.approx(0.02)

    def test_only_rics_present_on_both_sides_contribute(self):
        predicted = frame({DAYS[0]: {"A.HK": 0.01, "B.HK": 0.01}})
        realized = frame({DAYS[0]: {"A.HK": 0.03}})
        assert set(FactorAttribution().compute_difference(predicted, realized)["RIC"]) == {"A.HK"}

    def test_it_joins_on_date_as_well_as_ric(self):
        predicted = frame({DAYS[0]: {"A.HK": 0.01}, DAYS[1]: {"A.HK": 0.01}})
        realized = frame({DAYS[1]: {"A.HK": 0.03}})
        result = FactorAttribution().compute_difference(predicted, realized)
        assert len(result) == 1 and result["date"].iloc[0] == DAYS[1]

    def test_missing_columns_are_reported(self):
        with pytest.raises(ValueError, match="missing columns"):
            FactorAttribution().compute_difference(pd.DataFrame({"RIC": ["A.HK"]}), frame({DAYS[0]: {"A.HK": 0.01}}))

    def test_summary_reports_mean_and_dispersion(self):
        predicted = frame({DAYS[0]: {"A.HK": 0.0, "B.HK": 0.0}})
        realized = frame({DAYS[0]: {"A.HK": 0.01, "B.HK": 0.03}})
        difference = FactorAttribution().compute_difference(predicted, realized)
        summary = FactorAttribution().summarize_difference(difference)
        assert summary["mean_diff"] == pytest.approx(0.02)
        assert summary["std_diff"] == pytest.approx(np.std([0.01, 0.03], ddof=1))

    def test_an_empty_difference_summarizes_to_nans(self):
        summary = FactorAttribution().summarize_difference(pd.DataFrame(columns=["difference"]))
        assert np.isnan(summary["mean_diff"]) and np.isnan(summary["std_diff"])


class TestRegression:
    def _planted(self, loadings, seed=21):
        """difference = sum(loading_f * exposure_f) exactly, so the fit must recover them."""
        rng = np.random.default_rng(seed)
        exposures = {factor: rng.normal(size=len(RICS)) for factor in loadings}
        risk = pd.DataFrame({"RIC": RICS, "model": ["ASE2S"] * len(RICS), **exposures})
        difference = np.zeros(len(RICS))
        for factor, loading in loadings.items():
            difference += loading * exposures[factor]
        rows = [{"date": DAYS[0], "RIC": ric, "difference": d} for ric, d in zip(RICS, difference)]
        return pd.DataFrame(rows), risk

    def test_it_recovers_planted_factor_loadings(self, research_pit):
        difference, risk = self._planted({"mkt": 0.5, "size": -0.3, "ind_tech": 0.2})
        result = FactorAttribution(research_pit(risk=risk)).run_regression(difference, "HK", [DAYS[0]])
        assert result["barra_mkt_coefficient"] == pytest.approx(0.5, abs=1e-6)
        assert result["barra_size_coefficient"] == pytest.approx(-0.3, abs=1e-6)
        assert result["barra_ind_tech_coefficient"] == pytest.approx(0.2, abs=1e-6)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_industry_factors_serve_as_the_sector_breakdown(self, research_pit):
        difference, risk = self._planted({"mkt": 0.5, "ind_tech": 0.4, "ind_financials": -0.2})
        result = FactorAttribution(research_pit(risk=risk)).run_regression(difference, "HK", [DAYS[0]])
        assert {"barra_ind_tech_coefficient", "barra_ind_financials_coefficient"} <= set(result)

    def test_it_reports_observation_count_and_intercept(self, research_pit):
        difference, risk = self._planted({"mkt": 0.5})
        result = FactorAttribution(research_pit(risk=risk)).run_regression(difference, "HK", [DAYS[0]])
        assert result["n_obs"] == float(len(RICS)) and result["intercept"] == pytest.approx(0.0, abs=1e-6)

    def test_factor_names_are_not_hardcoded(self, research_pit):
        difference, risk = self._planted({"a_brand_new_factor": 0.7})
        result = FactorAttribution(research_pit(risk=risk)).run_regression(difference, "HK", [DAYS[0]])
        assert result["barra_a_brand_new_factor_coefficient"] == pytest.approx(0.7, abs=1e-6)

    def test_too_few_observations_degrade_gracefully(self, research_pit):
        difference = pd.DataFrame([{"date": DAYS[0], "RIC": RICS[0], "difference": 0.01}])
        risk = pd.DataFrame({"RIC": [RICS[0]], "model": ["ASE2S"], "mkt": [1.0]})
        result = FactorAttribution(research_pit(risk=risk)).run_regression(difference, "HK", [DAYS[0]])
        assert np.isnan(result["r_squared"]) and result["n_obs"] == 1.0

    def test_no_data_manager_means_no_attribution(self):
        difference = pd.DataFrame([{"date": DAYS[0], "RIC": r, "difference": 0.01} for r in RICS])
        result = FactorAttribution().run_regression(difference, "HK", [DAYS[0]])
        assert np.isnan(result["r_squared"]) and not any(k.startswith("barra_") for k in result)

    def test_a_constant_factor_is_dropped_rather_than_breaking_the_fit(self, research_pit):
        difference, risk = self._planted({"mkt": 0.5})
        risk["always_one"] = 1.0
        result = FactorAttribution(research_pit(risk=risk)).run_regression(difference, "HK", [DAYS[0]])
        assert "barra_always_one_coefficient" not in result
        assert result["barra_mkt_coefficient"] == pytest.approx(0.5, abs=1e-6)


class TestReport:
    def test_one_row_per_alpha(self, system_config):
        reporter = ReportGenerator(system_config.storage)
        report = reporter.assemble({"a1": {"r_squared": 0.18}, "a2": {"r_squared": 0.22}},
                                   {"a1": {"mean_diff": 0.002}, "a2": {"mean_diff": -0.001}})
        assert list(report["alpha_id"]) == ["a1", "a2"] and len(report) == 2

    def test_columns_run_summary_then_barra_then_intercept(self, system_config):
        reporter = ReportGenerator(system_config.storage)
        report = reporter.assemble(
            {"a1": {"r_squared": 0.1, "n_obs": 100.0, "intercept": 0.001,
                    "barra_size_coefficient": 0.2, "barra_mkt_coefficient": 0.05}},
            {"a1": {"mean_diff": 0.002, "std_diff": 0.015}})
        assert list(report.columns) == ["alpha_id", "mean_diff", "std_diff", "r_squared", "n_obs",
                                        "barra_mkt_coefficient", "barra_size_coefficient", "intercept"]

    def test_saving_refuses_to_clobber_without_overwrite(self, system_config, isolated_shared_drive):
        reporter = ReportGenerator(system_config.storage)
        report = reporter.assemble({"a1": {"r_squared": 0.1}}, {"a1": {"mean_diff": 0.0}})
        reporter.save(report, "hk_momentum", "20240131")
        assert (isolated_shared_drive / "post_trade_results" / "hk_momentum" / "20240131.parquet").is_file()
        with pytest.raises(FileExistsError):
            reporter.save(report, "hk_momentum", "20240131")
        reporter.save(report, "hk_momentum", "20240131", overwrite=True)
