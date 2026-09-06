import pytest

from alphaflow.post_trade_analyzer.runner import PostTradeAnalyzerRunner
from alphaflow.signal_builder.runner import SignalBuilderRunner

MARKET = "HK"
MOMENTUM, IMBALANCE, RETURN = "momentum_price_21d", "order_imbalance_1430", "open_to_close_1d"


@pytest.fixture
def scored(system_config, registry, raw_data_window, scoring_days, date_strings):
    raw_data_window(scoring_days)
    builder = SignalBuilderRunner(system_config, registry)
    dates = date_strings(scoring_days)
    for alpha_id in (MOMENTUM, IMBALANCE, RETURN):
        builder.build_multi_day_alpha(alpha_id, MARKET, dates, save=True)
    return dates


class TestPostTradePipeline:
    """stored scores + realised returns -> one attribution report per model."""

    def test_one_row_per_alpha_in_the_model(self, system_config, registry, scored, research_pit, mocker):
        mocker.patch("alphaflow.post_trade_analyzer.runner.PITDataManager.for_research", return_value=research_pit())
        report = PostTradeAnalyzerRunner(system_config, registry).run("hk_momentum", scored)["hk_momentum"]
        assert list(report["alpha_id"]) == sorted([MOMENTUM, IMBALANCE])

    def test_the_report_carries_summary_and_barra_columns(self, system_config, registry, scored, wide_risk_pit, mocker):
        mocker.patch("alphaflow.post_trade_analyzer.runner.PITDataManager.for_research", return_value=wide_risk_pit)
        report = PostTradeAnalyzerRunner(system_config, registry).run("hk_momentum", scored)["hk_momentum"]
        assert {"mean_diff", "std_diff", "r_squared", "n_obs", "intercept"} <= set(report.columns)
        assert any(c.startswith("barra_") and c.endswith("_coefficient") for c in report.columns)

    def test_industry_factors_are_present_as_the_sector_breakdown(self, system_config, registry, scored,
                                                                  wide_risk_pit, mocker):
        mocker.patch("alphaflow.post_trade_analyzer.runner.PITDataManager.for_research", return_value=wide_risk_pit)
        report = PostTradeAnalyzerRunner(system_config, registry).run("hk_momentum", scored)["hk_momentum"]
        assert {"barra_ind_tech_coefficient", "barra_ind_financials_coefficient"} <= set(report.columns)

    def test_more_factors_than_stocks_declines_to_attribute(self, system_config, registry, scored,
                                                            research_pit, mocker):
        # 5 universe RICs against 6 factors is rank-deficient - it must report NaN, not noise
        mocker.patch("alphaflow.post_trade_analyzer.runner.PITDataManager.for_research", return_value=research_pit())
        report = PostTradeAnalyzerRunner(system_config, registry).run("hk_momentum", scored)["hk_momentum"]
        assert report["r_squared"].isna().all()
        assert not any(c.startswith("barra_") for c in report.columns)

    def test_it_saves_to_the_spec_path(self, system_config, registry, scored, research_pit, mocker,
                                       isolated_shared_drive):
        mocker.patch("alphaflow.post_trade_analyzer.runner.PITDataManager.for_research", return_value=research_pit())
        PostTradeAnalyzerRunner(system_config, registry).run("hk_momentum", scored)
        assert list((isolated_shared_drive / "post_trade_results" / "hk_momentum").glob("*.parquet"))

    def test_missing_returns_are_reported(self, system_config, registry, scored):
        with pytest.raises(FileNotFoundError, match="no stored returns"):
            PostTradeAnalyzerRunner(system_config, registry).run("hk_momentum", ["20200101"])

    def test_an_empty_date_list_is_rejected(self, system_config, registry):
        with pytest.raises(ValueError, match="dates must not be empty"):
            PostTradeAnalyzerRunner(system_config, registry).run("hk_momentum", [])
