import numpy as np
import pandas as pd

from alphaflow.strategy_backtester.score_loader import RIC_COLUMN, ScoreLoader

OLS_MEASURES = ["r_squared", "t_stat", "coefficient", "intercept", "n_stocks"]


class CrossSectionalOLS:
    """Pooled cross-sectional OLS over the whole period at once - every (date, RIC)
    observation in one regression of forward_return on alpha_score."""

    def fit(self, alpha_scores: pd.DataFrame, return_scores: pd.DataFrame) -> dict[str, float]:
        aligned = ScoreLoader.align(alpha_scores, return_scores)
        if len(aligned) < 3 or aligned["score"].nunique() < 2:
            return dict.fromkeys(OLS_MEASURES, np.nan)
        import statsmodels.api as sm
        design = sm.add_constant(aligned["score"].to_numpy(dtype=float), has_constant="add")
        model = sm.OLS(aligned["forward_return"].to_numpy(dtype=float), design).fit()
        return {
            "r_squared": float(model.rsquared),
            "t_stat": float(model.tvalues[1]),
            "coefficient": float(model.params[1]),
            "intercept": float(model.params[0]),
            "n_stocks": float(aligned[RIC_COLUMN].nunique()),
        }
