from pydantic import Field

from alphaflow.core.config.alpha_config.base_config import BaseAlphaConfig


class MidAlphaConfig(BaseAlphaConfig):
    """Interval-based intraday. No concrete mid alpha exists yet - this carries only the
    sampling interval until the first implementation defines what else it needs."""

    interval_minutes: int = Field(default=5, ge=1)
