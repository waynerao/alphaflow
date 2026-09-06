import numpy as np
import pandas as pd

from alphaflow.strategy_backtester.score_loader import DATE_COLUMN, ScoreLoader

N_DECILES = 10


class DecileAnalyzer:
    """Sorts each day's cross-section into deciles by score and averages the forward return
    in each. Decile 1 is the lowest score, decile 10 the highest."""

    def compute_decile_returns(self, alpha_scores: pd.DataFrame, return_scores: pd.DataFrame,
                               n_deciles: int = N_DECILES) -> pd.DataFrame:
        aligned = ScoreLoader.align(alpha_scores, return_scores)
        if aligned.empty:
            return pd.DataFrame(columns=["decile", "mean_return"])
        frames = []
        for day, group in aligned.groupby(DATE_COLUMN):
            if len(group) < n_deciles:
                continue    # too few names to fill the buckets; the day contributes nothing
            labelled = group.assign(decile=self._assign_deciles(group["score"], n_deciles))
            frames.append(labelled.groupby("decile", observed=True)["forward_return"].mean().rename(day))
        if not frames:
            return pd.DataFrame(columns=["decile", "mean_return"])
        per_day = pd.concat(frames, axis=1)
        return per_day.mean(axis=1).rename("mean_return").reset_index()

    @classmethod
    def _assign_deciles(cls, scores: pd.Series, n_deciles: int) -> pd.Series:
        """Rank-then-cut rather than qcut on raw values, so ties and clustered scores still
        produce populated buckets."""
        ranks = scores.rank(method="first", pct=True)
        return np.minimum((ranks * n_deciles).apply(np.ceil).astype(int), n_deciles)

    def compute_spread(self, decile_returns: pd.DataFrame, n_deciles: int = N_DECILES) -> float:
        if decile_returns.empty:
            return np.nan
        indexed = decile_returns.set_index("decile")["mean_return"]
        if n_deciles not in indexed.index or 1 not in indexed.index:
            return np.nan
        return float(indexed.loc[n_deciles] - indexed.loc[1])
