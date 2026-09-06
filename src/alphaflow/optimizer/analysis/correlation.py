import numpy as np
import pandas as pd


class CorrelationAnalyzer:
    """Correlation feeds the covariance matrix - correlated alphas are down-weighted by the
    optimizer rather than excluded by a threshold."""

    def compute(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        return daily_returns.corr()

    def average_correlation(self, corr_matrix: pd.DataFrame, alpha_id: str) -> float:
        """Mean correlation of one alpha against every other - its own diagonal excluded."""
        if alpha_id not in corr_matrix.index or len(corr_matrix) < 2:
            return np.nan
        others = corr_matrix.loc[alpha_id].drop(index=alpha_id)
        return float(others.mean()) if not others.empty else np.nan
