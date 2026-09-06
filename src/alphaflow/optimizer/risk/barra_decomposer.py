from datetime import date

import pandas as pd

from alphaflow.strategy_backtester.risk.barra_exposure import BarraExposure


class BarraDecomposer:
    """Per-alpha Barra exposures, and the weighted combination of them.

    Factor names are whatever the risk model returns (decision D7), so a model with extra
    industry factors needs no code change here.
    """

    def __init__(self, data=None) -> None:
        self.exposure = BarraExposure(data) if data is not None else None

    def compute_standalone(self, alpha_scores: pd.DataFrame, market: str, dates: list[date]) -> dict[str, float]:
        return self.exposure.compute(alpha_scores, market, dates) if self.exposure is not None else {}

    def compute_combined(self, standalone_exposures: dict[str, dict[str, float]], weights: pd.Series) -> dict[str, float]:
        """Portfolio exposure = weighted sum of the per-alpha exposures. An alpha missing a
        factor contributes nothing to it rather than dragging the sum to NaN."""
        if not standalone_exposures:
            return {}
        frame = pd.DataFrame(standalone_exposures)                 # rows = factors, cols = alpha_ids
        aligned = weights.reindex(frame.columns).fillna(0.0)
        combined = frame.fillna(0.0).mul(aligned, axis=1).sum(axis=1)
        return {factor: float(value) for factor, value in combined.items()}
