from datetime import date

import pandas as pd

from alphaflow.core.config.system_config import StorageConfig
from alphaflow.core.utils.storage import StoragePaths

RIC_COLUMN = "RIC"
COMPOSITE_COLUMN = "composite"


class ParquetWriter:
    """Archives a model's individual scores plus its composite to
    alpha_scores/{model_id}/YYYYMMDD.parquet. Written only when the model config asks for it."""

    def __init__(self, storage_config: StorageConfig) -> None:
        self.paths = StoragePaths(storage_config)

    def write(self, model_id: str, individual_scores: dict[str, pd.Series], composite_score: pd.Series, date: date) -> str:
        frame = self.build_frame(individual_scores, composite_score)
        path = StoragePaths.ensure_parent(self.paths.model_score(model_id, date))
        frame.to_parquet(path, index=False)
        return str(path)

    @classmethod
    def build_frame(cls, individual_scores: dict[str, pd.Series], composite_score: pd.Series) -> pd.DataFrame:
        """One row per RIC: a column per alpha, plus the composite. Keeping the components
        alongside the total is what makes a production score reconstructable after the fact."""
        frame = pd.DataFrame({COMPOSITE_COLUMN: composite_score})
        for alpha_id, series in (individual_scores or {}).items():
            frame[alpha_id] = series.reindex(frame.index)
        ordered = [COMPOSITE_COLUMN] + [c for c in frame.columns if c != COMPOSITE_COLUMN]
        return frame[ordered].rename_axis(RIC_COLUMN).reset_index()
