from pydantic import Field, field_validator

from alphaflow.core.config.alpha_config.base_config import BaseAlphaConfig
from alphaflow.core.config.system_config import validate_hhmm


class HighAlphaConfig(BaseAlphaConfig):
    """Tick-level intraday. Fields kept minimal and driven by the first concrete
    implementation (order_imbalance_1430); extend as further high alphas land."""

    snapshot_time: str = Field(description="HH:MM order-book snapshot this alpha reads")
    depth_levels: int = Field(default=5, ge=1, le=10)

    _check_snapshot_time = field_validator("snapshot_time")(staticmethod(validate_hhmm))
