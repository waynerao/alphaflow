"""Template for a new LowAlpha.

Copy to <alpha_id>.py, copy configs/alphas/low/_template.toml to <alpha_id>.toml, then add a
[[alphas]] entry to configs/registry.toml. Those three files are the whole checklist.

Do not import this module - it is a reference, and its class is intentionally not registered.
"""

from datetime import date

import pandas as pd

from alphaflow.core.alpha_base.low_alpha import LowAlpha


class TemplateAlpha(LowAlpha):
    SHIFT: float = 0.0
    SCALE: float | None = None    # None -> divide by the cross-sectional std

    def compute(self, market: str, as_of_date: date) -> pd.Series:
        # self.working_data is the RIC-joined, null-dropped frame of required_inputs.
        # Return one value per RIC; drop RICs you cannot score rather than zero-filling them.
        df = self.working_data
        signal = df.groupby("RIC")["close"].last()
        signal.name = self.alpha_id
        return signal

    def normalize(self, raw_signal: pd.Series) -> pd.Series:
        # Delete this method entirely to keep the base class's identity normalisation.
        scale = self.SCALE if self.SCALE is not None else raw_signal.std()
        if not scale or pd.isna(scale):
            return pd.Series(0.0, index=raw_signal.index, name=raw_signal.name)
        return (raw_signal - self.SHIFT) / scale
