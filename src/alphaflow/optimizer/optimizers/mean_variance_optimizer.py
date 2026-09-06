import numpy as np
import pandas as pd
from scipy.optimize import minimize

from alphaflow.core.config.optimizer_config import MeanVarianceConfig
from alphaflow.core.integrations.logging_setup import setup_logger
from alphaflow.core.utils.stats import safe_ratio, sample_std

TRADING_DAYS_PER_YEAR = 252
log = setup_logger(__name__)


class MeanVarianceOptimizer:
    """maximize  w'mu - risk_aversion * w'Sigma w
       s.t.      min_weight <= w_i <= max_weight,  sum(w) = 1

    Solved as a minimisation of the negated objective via SLSQP.
    """

    def __init__(self, config: MeanVarianceConfig) -> None:
        self.config = config

    def optimize(self, daily_returns: pd.DataFrame) -> pd.Series:
        alpha_ids = list(daily_returns.columns)
        n = len(alpha_ids)
        if n == 0:
            raise ValueError("daily_returns has no alpha columns")
        if n == 1:
            # Nothing to allocate between. The per-alpha cap is a diversification constraint
            # that only bites across several alphas, so it does not veto a single-alpha model.
            return pd.Series([1.0], index=alpha_ids, name="weight")
        if not self.config.feasible_for(n):
            raise ValueError(f"weight bounds [{self.config.min_single_alpha_weight}, {self.config.max_single_alpha_weight}] "
                             f"cannot sum to 1 across {n} alpha(s)")

        mu = daily_returns.mean().to_numpy(dtype=float)
        sigma = daily_returns.cov().to_numpy(dtype=float)
        bounds = [(self.config.min_single_alpha_weight, self.config.max_single_alpha_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        start = self._feasible_start(n)

        result = minimize(self._negated_objective, start, args=(mu, sigma, self.config.risk_aversion),
                          method="SLSQP", bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-12})
        if not result.success:
            # An unconverged solve would silently ship arbitrary weights, so fall back to
            # the equal-weight portfolio and say so loudly.
            log.error(f"SLSQP did not converge ({result.message}); falling back to equal weights")
            return pd.Series(start, index=alpha_ids, name="weight")
        weights = np.clip(result.x, self.config.min_single_alpha_weight, self.config.max_single_alpha_weight)
        return pd.Series(weights / weights.sum(), index=alpha_ids, name="weight")

    def _feasible_start(self, n: int) -> np.ndarray:
        """Equal weights, pulled inside the box if the bounds forbid 1/n."""
        return np.clip(np.full(n, 1.0 / n), self.config.min_single_alpha_weight, self.config.max_single_alpha_weight)

    @classmethod
    def _negated_objective(cls, w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, risk_aversion: float) -> float:
        return -(w @ mu - risk_aversion * (w @ sigma @ w))

    def compute_portfolio_stats(self, daily_returns: pd.DataFrame, weights: pd.Series,
                                periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict[str, float]:
        aligned = weights.reindex(daily_returns.columns).fillna(0.0)
        portfolio = daily_returns.mul(aligned, axis=1).sum(axis=1)
        mean = float(portfolio.mean())
        std = sample_std(portfolio)
        return {"standalone_mean_return": mean, "standalone_std": std,
                "standalone_sharpe": safe_ratio(mean, std, np.sqrt(periods_per_year))}
