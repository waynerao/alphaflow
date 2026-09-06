from alphaflow.core.alpha_base.base_alpha import BaseAlpha


class HighAlpha(BaseAlpha):
    """Tick-level intraday - data updates on order-book change.

    compute() sees a _working_data frame that still carries its intraday timestamp column;
    reducing it to one value per RIC (typically at config.snapshot_time) is the alpha's job.
    """

    EXPECTED_ALPHA_TYPE = "high"
