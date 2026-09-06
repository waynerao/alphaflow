import numpy as np
import pandas as pd

from alphaflow.core.utils.stats import safe_ratio, sample_std
from alphaflow.strategy_backtester.score_loader import DATE_COLUMN, RIC_COLUMN

TRADING_DAYS_PER_YEAR = 252


class PerformanceMetrics:
    def compute_sharpe(self, ic_series: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
        """Annualised Sharpe of the daily IC series - the spec's proxy for strategy Sharpe."""
        if len(ic_series) < 2:
            return np.nan
        return safe_ratio(ic_series.mean(), sample_std(ic_series), np.sqrt(periods_per_year))

    def compute_max_drawdown(self, ic_series: pd.Series) -> float:
        """Deepest peak-to-trough fall in cumulative IC. Returned as a negative number
        (0.0 when the curve never retraces)."""
        if ic_series.empty:
            return np.nan
        cumulative = ic_series.cumsum()
        return float((cumulative - cumulative.cummax()).min())

    def compute_turnover(self, alpha_scores: pd.DataFrame) -> float:
        """Mean absolute day-on-day change in cross-sectional percentile rank, over the RICs
        present on both days. 0 = a completely static book, ~0.33 = random reshuffling."""
        if alpha_scores.empty or DATE_COLUMN not in alpha_scores.columns:
            return np.nan
        ranked = alpha_scores.copy()
        ranked["_rank"] = ranked.groupby(DATE_COLUMN)["score"].rank(pct=True)
        matrix = ranked.pivot_table(index=DATE_COLUMN, columns=RIC_COLUMN, values="_rank").sort_index()
        if len(matrix) < 2:
            return np.nan
        changes = matrix.diff().abs()
        daily = changes.mean(axis=1, skipna=True).dropna()
        return float(daily.mean()) if not daily.empty else np.nan
