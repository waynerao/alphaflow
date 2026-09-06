import numpy as np
import pandas as pd

# Floating-point noise floor for a standard deviation. A constant series does not reduce to
# exactly 0.0 under ddof=1, so an unguarded mean/std produces a vast ratio that reads as a
# spectacular result rather than the degenerate one it is.
MIN_STD = 1e-12


def is_degenerate(std: float) -> bool:
    return std is None or not np.isfinite(std) or abs(std) < MIN_STD


def safe_ratio(numerator: float, std: float, scale: float = 1.0) -> float:
    """numerator / std * scale, or NaN when std is degenerate."""
    if is_degenerate(std):
        return np.nan
    return float(numerator / std * scale)


def sample_std(values: pd.Series) -> float:
    """Sample standard deviation, or NaN when there is nothing to disperse."""
    return float(values.std(ddof=1)) if len(values) > 1 else np.nan
