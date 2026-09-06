from alphaflow.core.alpha_base.base_alpha import BaseAlpha


class MidAlpha(BaseAlpha):
    """Interval-based intraday - sampled on a fixed config.interval_minutes grid."""

    EXPECTED_ALPHA_TYPE = "mid"
