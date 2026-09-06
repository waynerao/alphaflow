from datetime import date

import pandas as pd

from alphaflow.core.config.system_config import StorageConfig
from alphaflow.signal_builder.storage.score_writer import ScoreWriter

RIC_COLUMN = "RIC"
DATE_COLUMN = "date"


class ScoreLoader:
    """Reads pre-computed scores off the shared drive. Shared with the optimizer.

    The backtester never computes an alpha - signal_builder owns that. This is the read side
    of the file-based hand-off in spec section 12.
    """

    def __init__(self, storage_config: StorageConfig) -> None:
        self.writer = ScoreWriter(storage_config)

    def load_alpha_scores(self, market: str, alpha_id: str, dates: list[str | date]) -> pd.DataFrame:
        return self.writer.read_dates(market, alpha_id, dates)

    def load_return_scores(self, market: str, rtn_alpha_id: str, dates: list[str | date]) -> pd.DataFrame:
        """Same storage layout - an RtnAlpha's realised return is just another alpha score."""
        return self.writer.read_dates(market, rtn_alpha_id, dates)

    def available_dates(self, market: str, alpha_id: str) -> list[str]:
        return self.writer.available_dates(market, alpha_id)

    @classmethod
    def align(cls, alpha_scores: pd.DataFrame, return_scores: pd.DataFrame) -> pd.DataFrame:
        """Inner-join scores to returns on (date, RIC). Columns: date, RIC, score, forward_return."""
        for name, frame in (("alpha_scores", alpha_scores), ("return_scores", return_scores)):
            missing = [c for c in (DATE_COLUMN, RIC_COLUMN, "score") if c not in frame.columns]
            if missing:
                raise ValueError(f"{name} missing columns {missing}; got {list(frame.columns)}")
        left = alpha_scores[[DATE_COLUMN, RIC_COLUMN, "score"]]
        right = return_scores[[DATE_COLUMN, RIC_COLUMN, "score"]].rename(columns={"score": "forward_return"})
        return left.merge(right, on=[DATE_COLUMN, RIC_COLUMN], how="inner").dropna().reset_index(drop=True)
