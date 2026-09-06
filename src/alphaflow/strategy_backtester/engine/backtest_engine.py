from datetime import date

import pandas as pd

from alphaflow.strategy_backtester.metrics.decile_analyzer import DecileAnalyzer
from alphaflow.strategy_backtester.metrics.ic_calculator import ICCalculator
from alphaflow.strategy_backtester.metrics.performance_metrics import PerformanceMetrics
from alphaflow.strategy_backtester.regression.cross_sectional_ols import CrossSectionalOLS
from alphaflow.strategy_backtester.risk.barra_exposure import BarraExposure

# Fixed leading order of the result table's measure rows; discovered Barra rows follow.
MEASURE_ORDER = [
    "mean_IC", "std_IC", "ICIR", "sharpe", "turnover", "max_drawdown", "decile_spread",
    "r_squared", "t_stat", "coefficient", "intercept", "n_stocks",
]


class BacktestEngine:
    """Orchestrates every metric for one alpha against one rtn_alpha_id."""

    def __init__(self, data=None) -> None:
        self.ic = ICCalculator()
        self.performance = PerformanceMetrics()
        self.deciles = DecileAnalyzer()
        self.ols = CrossSectionalOLS()
        self.barra = BarraExposure(data) if data is not None else None

    def run(self, alpha_scores: pd.DataFrame, return_scores: pd.DataFrame, market: str, dates: list[date]) -> dict[str, float]:
        ic_series = self.ic.compute_daily_ic(alpha_scores, return_scores)
        decile_returns = self.deciles.compute_decile_returns(alpha_scores, return_scores)
        results: dict[str, float] = {}
        results.update(self.ic.summarize(ic_series))
        results["sharpe"] = self.performance.compute_sharpe(ic_series)
        results["turnover"] = self.performance.compute_turnover(alpha_scores)
        results["max_drawdown"] = self.performance.compute_max_drawdown(ic_series)
        results["decile_spread"] = self.deciles.compute_spread(decile_returns)
        results.update(self.ols.fit(alpha_scores, return_scores))
        if self.barra is not None:
            results.update(self.barra.compute(alpha_scores, market, dates))
        return results
