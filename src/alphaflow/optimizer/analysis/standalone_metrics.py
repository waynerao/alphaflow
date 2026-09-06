import numpy as np
import pandas as pd

from alphaflow.core.utils.stats import safe_ratio, sample_std

TRADING_DAYS_PER_YEAR = 252


class StandaloneMetrics:
    """Per-alpha statistics computed on its own daily return series, before any combination."""

    def compute(self, daily_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict[str, float]:
        if daily_returns.empty:
            return {"standalone_sharpe": np.nan, "standalone_std": np.nan, "standalone_mean_return": np.nan}
        mean = float(daily_returns.mean())
        std = sample_std(daily_returns)
        return {"standalone_sharpe": safe_ratio(mean, std, np.sqrt(periods_per_year)),
                "standalone_std": std, "standalone_mean_return": mean}

    def compute_all(self, daily_returns: pd.DataFrame) -> dict[str, dict[str, float]]:
        return {column: self.compute(daily_returns[column]) for column in daily_returns.columns}
