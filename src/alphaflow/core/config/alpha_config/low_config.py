from pydantic import Field

from alphaflow.core.config.alpha_config.base_config import BaseAlphaConfig


class LowAlphaConfig(BaseAlphaConfig):
    """Daily / multi-day. Fields driven by the first concrete implementation
    (momentum_price_21d)."""

    lookback_days: int = Field(ge=1)
    skip_days: int = Field(default=0, ge=0)
