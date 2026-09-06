from datetime import date, timedelta

import pandas as pd
import pytest

from alphaflow.optimizer.analysis.correlation import CorrelationAnalyzer
from alphaflow.optimizer.analysis.standalone_metrics import StandaloneMetrics
from alphaflow.optimizer.returns.strategy_return_builder import StrategyReturnBuilder
from alphaflow.strategy_backtester.score_loader import ScoreLoader

DAYS = [date(2024, 1, 1) + timedelta(days=d) for d in range(5)]


def frame(per_day):
    rows = []
    for day, values in per_day.items():
        for ric, value in values.items():
            rows.append({"date": day, "RIC": ric, "raw_signal": value, "score": value})
    return pd.DataFrame(rows)


class TestDailyReturn:
    def test_it_is_the_sum_of_score_times_return(self):
        scores = frame({DAYS[0]: {"A.HK": 2.0, "B.HK": 3.0}})
        returns = frame({DAYS[0]: {"A.HK": 0.01, "B.HK": 0.02}})
        assert StrategyReturnBuilder().build(scores, returns).loc[DAYS[0]] == pytest.approx(2 * 0.01 + 3 * 0.02)

    def test_it_is_not_rescaled_to_dollar_neutral(self):
        scores = frame({DAYS[0]: {"A.HK": 10.0, "B.HK": 10.0}})
        returns = frame({DAYS[0]: {"A.HK": 0.01, "B.HK": 0.01}})
        assert StrategyReturnBuilder().build(scores, returns).loc[DAYS[0]] == pytest.approx(0.2)

    def test_unmatched_rics_contribute_nothing(self):
        scores = frame({DAYS[0]: {"A.HK": 2.0, "GHOST.HK": 100.0}})
        returns = frame({DAYS[0]: {"A.HK": 0.01}})
        assert StrategyReturnBuilder().build(scores, returns).loc[DAYS[0]] == pytest.approx(0.02)

    def test_one_value_per_date_in_order(self):
        scores = frame({day: {"A.HK": 1.0} for day in reversed(DAYS)})
        returns = frame({day: {"A.HK": 0.01} for day in DAYS})
        result = StrategyReturnBuilder().build(scores, returns)
        assert len(result) == len(DAYS) and list(result.index) == DAYS

    def test_no_overlap_yields_an_empty_series(self):
        scores = frame({DAYS[0]: {"A.HK": 1.0}})
        returns = frame({DAYS[1]: {"A.HK": 0.01}})
        assert StrategyReturnBuilder().build(scores, returns).empty


class TestBuildForModel:
    @pytest.fixture
    def loader(self, system_config):
        return ScoreLoader(system_config.storage)

    def _store(self, loader, market, alpha_id, per_day):
        for day, values in per_day.items():
            rows = pd.DataFrame({"RIC": list(values), "raw_signal": list(values.values()), "score": list(values.values())})
            loader.writer.write(rows, market, alpha_id, day)

    def test_one_column_per_alpha_indexed_by_date(self, loader):
        self._store(loader, "HK", "a1", {day: {"A.HK": 1.0} for day in DAYS})
        self._store(loader, "HK", "a2", {day: {"A.HK": 2.0} for day in DAYS})
        self._store(loader, "HK", "r1", {day: {"A.HK": 0.01} for day in DAYS})
        dates = [d.strftime("%Y%m%d") for d in DAYS]
        result = StrategyReturnBuilder().build_for_model(["a1", "a2"], "HK", "r1", dates, loader)
        assert list(result.columns) == ["a1", "a2"] and len(result) == len(DAYS)
        assert result["a2"].iloc[0] == pytest.approx(0.02)

    def test_dates_missing_for_one_alpha_are_dropped_from_all(self, loader):
        self._store(loader, "HK", "a1", {day: {"A.HK": 1.0} for day in DAYS})
        self._store(loader, "HK", "a2", {day: {"A.HK": 1.0} for day in DAYS[:3]})
        self._store(loader, "HK", "r1", {day: {"A.HK": 0.01} for day in DAYS})
        dates = [d.strftime("%Y%m%d") for d in DAYS]
        assert len(StrategyReturnBuilder().build_for_model(["a1", "a2"], "HK", "r1", dates, loader)) == 3

    def test_a_missing_alpha_is_reported(self, loader):
        self._store(loader, "HK", "r1", {day: {"A.HK": 0.01} for day in DAYS})
        dates = [d.strftime("%Y%m%d") for d in DAYS]
        with pytest.raises(FileNotFoundError, match="no stored scores for ghost"):
            StrategyReturnBuilder().build_for_model(["ghost"], "HK", "r1", dates, loader)

    def test_a_missing_return_alpha_is_reported(self, loader):
        dates = [d.strftime("%Y%m%d") for d in DAYS]
        with pytest.raises(FileNotFoundError, match="no stored return scores"):
            StrategyReturnBuilder().build_for_model(["a1"], "HK", "ghost", dates, loader)


class TestCorrelationAndStandalone:
    def test_identical_series_correlate_perfectly(self):
        returns = pd.DataFrame({"a": [0.01, 0.02, 0.03], "b": [0.01, 0.02, 0.03]})
        assert CorrelationAnalyzer().compute(returns).loc["a", "b"] == pytest.approx(1.0)

    def test_inverted_series_correlate_negatively(self):
        returns = pd.DataFrame({"a": [0.01, 0.02, 0.03], "b": [0.03, 0.02, 0.01]})
        assert CorrelationAnalyzer().compute(returns).loc["a", "b"] == pytest.approx(-1.0)

    def test_average_correlation_excludes_the_diagonal(self):
        returns = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0], "c": [3.0, 2.0, 1.0]})
        analyzer = CorrelationAnalyzer()
        assert analyzer.average_correlation(analyzer.compute(returns), "a") == pytest.approx(0.0)

    def test_a_single_alpha_has_no_average_correlation(self):
        analyzer = CorrelationAnalyzer()
        matrix = analyzer.compute(pd.DataFrame({"a": [1.0, 2.0]}))
        assert pd.isna(analyzer.average_correlation(matrix, "a"))

    def test_standalone_metrics_cover_all_columns(self):
        returns = pd.DataFrame({"a": [0.01, 0.02, 0.03], "b": [-0.01, 0.0, 0.01]})
        result = StandaloneMetrics().compute_all(returns)
        assert set(result) == {"a", "b"}
        assert result["a"]["standalone_mean_return"] == pytest.approx(0.02)

    def test_a_flat_series_has_no_sharpe(self):
        assert pd.isna(StandaloneMetrics().compute(pd.Series([0.01] * 5))["standalone_sharpe"])
