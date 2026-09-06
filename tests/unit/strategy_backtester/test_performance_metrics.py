from datetime import date

import numpy as np
import pandas as pd
import pytest

from alphaflow.strategy_backtester.metrics.decile_analyzer import DecileAnalyzer
from alphaflow.strategy_backtester.metrics.performance_metrics import TRADING_DAYS_PER_YEAR, PerformanceMetrics

DAYS = [date(2024, 1, d) for d in range(1, 11)]


@pytest.fixture
def metrics():
    return PerformanceMetrics()


class TestSharpe:
    def test_annualizes_by_the_square_root_of_the_period_count(self, metrics):
        series = pd.Series([0.01, 0.02, 0.03, 0.04])
        expected = series.mean() / series.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        assert metrics.compute_sharpe(series) == pytest.approx(expected)

    def test_a_constant_series_has_no_sharpe(self, metrics):
        assert np.isnan(metrics.compute_sharpe(pd.Series([0.01] * 5)))

    def test_a_single_point_has_no_sharpe(self, metrics):
        assert np.isnan(metrics.compute_sharpe(pd.Series([0.01])))

    def test_a_negative_mean_gives_a_negative_sharpe(self, metrics):
        assert metrics.compute_sharpe(pd.Series([-0.01, -0.02, -0.03])) < 0


class TestMaxDrawdown:
    def test_a_monotonically_rising_curve_never_draws_down(self, metrics):
        assert metrics.compute_max_drawdown(pd.Series([0.1, 0.1, 0.1])) == pytest.approx(0.0)

    def test_the_deepest_peak_to_trough_fall_is_returned(self, metrics):
        # cumulative: 1.0, 0.4, 0.1, 0.6  -> peak 1.0, trough 0.1
        series = pd.Series([1.0, -0.6, -0.3, 0.5])
        assert metrics.compute_max_drawdown(series) == pytest.approx(-0.9)

    def test_the_result_is_never_positive(self, metrics):
        assert metrics.compute_max_drawdown(pd.Series([-0.1, 0.2, -0.3])) <= 0

    def test_an_empty_series_yields_nan(self, metrics):
        assert np.isnan(metrics.compute_max_drawdown(pd.Series(dtype=float)))


class TestTurnover:
    def _scores(self, per_day):
        rows = []
        for day, values in per_day.items():
            for ric, value in values.items():
                rows.append({"date": day, "RIC": ric, "score": value})
        return pd.DataFrame(rows)

    def test_an_unchanged_book_has_zero_turnover(self, metrics):
        frame = self._scores({day: {"A.HK": 1.0, "B.HK": 2.0, "C.HK": 3.0} for day in DAYS[:3]})
        assert metrics.compute_turnover(frame) == pytest.approx(0.0)

    def test_a_fully_reversed_ranking_turns_over_the_most(self, metrics):
        frame = self._scores({DAYS[0]: {"A.HK": 1.0, "B.HK": 2.0, "C.HK": 3.0},
                              DAYS[1]: {"A.HK": 3.0, "B.HK": 2.0, "C.HK": 1.0}})
        assert metrics.compute_turnover(frame) == pytest.approx((2 / 3 + 0 + 2 / 3) / 3)

    def test_a_single_day_has_no_turnover(self, metrics):
        assert np.isnan(metrics.compute_turnover(self._scores({DAYS[0]: {"A.HK": 1.0, "B.HK": 2.0}})))

    def test_an_empty_frame_yields_nan(self, metrics):
        assert np.isnan(metrics.compute_turnover(pd.DataFrame()))


class TestDecileAnalyzer:
    def _frames(self, n=100):
        rows_a, rows_r = [], []
        for day in DAYS[:3]:
            for i in range(n):
                # forward return increases with score, so decile 10 must beat decile 1
                rows_a.append({"date": day, "RIC": f"{i:04d}.HK", "score": float(i)})
                rows_r.append({"date": day, "RIC": f"{i:04d}.HK", "score": float(i) * 0.01})
        return pd.DataFrame(rows_a), pd.DataFrame(rows_r)

    def test_ten_deciles_are_produced(self):
        alpha, returns = self._frames()
        assert len(DecileAnalyzer().compute_decile_returns(alpha, returns)) == 10

    def test_decile_returns_are_monotone_for_a_monotone_signal(self):
        alpha, returns = self._frames()
        result = DecileAnalyzer().compute_decile_returns(alpha, returns).sort_values("decile")
        assert result["mean_return"].is_monotonic_increasing

    def test_spread_is_top_minus_bottom(self):
        alpha, returns = self._frames()
        analyzer = DecileAnalyzer()
        result = analyzer.compute_decile_returns(alpha, returns).set_index("decile")["mean_return"]
        assert analyzer.compute_spread(result.reset_index()) == pytest.approx(result.loc[10] - result.loc[1])

    def test_a_day_with_fewer_names_than_deciles_is_skipped(self):
        alpha = pd.DataFrame([{"date": DAYS[0], "RIC": f"{i}.HK", "score": float(i)} for i in range(5)])
        returns = alpha.copy()
        assert DecileAnalyzer().compute_decile_returns(alpha, returns).empty

    def test_an_empty_input_yields_nan_spread(self):
        assert np.isnan(DecileAnalyzer().compute_spread(pd.DataFrame(columns=["decile", "mean_return"])))
