from alphaflow.core.alpha_base.base_alpha import BaseAlpha


class LowAlpha(BaseAlpha):
    """Daily / multi-day. _working_data spans the lookback window, so compute() reduces a
    per-RIC history down to one value."""

    EXPECTED_ALPHA_TYPE = "low"
