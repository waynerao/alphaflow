import numpy as np
import pandas as pd

from alphaflow.core.utils.stats import safe_ratio, sample_std
from alphaflow.strategy_backtester.score_loader import DATE_COLUMN, ScoreLoader

MIN_STOCKS_FOR_IC = 3


class ICCalculator:
    """Daily cross-sectional Spearman rank correlation of score against forward return."""

    def compute_daily_ic(self, alpha_scores: pd.DataFrame, return_scores: pd.DataFrame) -> pd.Series:
        aligned = ScoreLoader.align(alpha_scores, return_scores)
        if aligned.empty:
            return pd.Series(dtype=float, name="ic")
        daily = aligned.groupby(DATE_COLUMN).apply(self._spearman, include_groups=False)
        return daily.dropna().rename("ic")

    @classmethod
    def _spearman(cls, group: pd.DataFrame) -> float:
        # A handful of names cannot produce a meaningful rank correlation, and a constant
        # score has no ranking at all - both are dropped rather than reported as zero
        if len(group) < MIN_STOCKS_FOR_IC or group["score"].nunique() < 2 or group["forward_return"].nunique() < 2:
            return np.nan
        return group["score"].corr(group["forward_return"], method="spearman")

    def summarize(self, ic_series: pd.Series) -> dict[str, float]:
        if ic_series.empty:
            return {"mean_IC": np.nan, "std_IC": np.nan, "ICIR": np.nan}
        mean_ic = float(ic_series.mean())
        std_ic = sample_std(ic_series)
        return {"mean_IC": mean_ic, "std_IC": std_ic, "ICIR": safe_ratio(mean_ic, std_ic)}
