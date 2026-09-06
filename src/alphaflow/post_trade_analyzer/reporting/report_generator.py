from datetime import date

import pandas as pd

from alphaflow.core.config.system_config import StorageConfig
from alphaflow.core.utils.storage import StoragePaths
from alphaflow.post_trade_analyzer.attribution.factor_attribution import COEFFICIENT_SUFFIX
from alphaflow.strategy_backtester.risk.barra_exposure import BARRA_PREFIX

ALPHA_ID_COLUMN = "alpha_id"
LEADING_COLUMNS = ["mean_diff", "std_diff", "r_squared", "n_obs"]
TRAILING_COLUMNS = ["intercept"]


class ReportGenerator:
    """One combined report per model: a row per alpha, Barra coefficients as columns."""

    def __init__(self, storage_config: StorageConfig) -> None:
        self.paths = StoragePaths(storage_config)

    def assemble(self, per_alpha_results: dict[str, dict[str, float]],
                 per_alpha_summary: dict[str, dict[str, float]]) -> pd.DataFrame:
        if not per_alpha_results and not per_alpha_summary:
            return pd.DataFrame(columns=[ALPHA_ID_COLUMN])
        rows = {alpha_id: {**per_alpha_summary.get(alpha_id, {}), **per_alpha_results.get(alpha_id, {})}
                for alpha_id in {*per_alpha_results, *per_alpha_summary}}
        table = pd.DataFrame.from_dict(rows, orient="index").sort_index()
        return table[self._ordered_columns(table.columns)].rename_axis(ALPHA_ID_COLUMN).reset_index()

    @classmethod
    def _ordered_columns(cls, columns: pd.Index) -> list[str]:
        """Summary stats, then Barra coefficients alphabetically, then the intercept last -
        so two reports are always column-comparable."""
        leading = [c for c in LEADING_COLUMNS if c in columns]
        trailing = [c for c in TRAILING_COLUMNS if c in columns]
        barra = sorted(c for c in columns if c.startswith(BARRA_PREFIX) and c.endswith(COEFFICIENT_SUFFIX))
        remainder = [c for c in columns if c not in leading and c not in trailing and c not in barra]
        return leading + barra + remainder + trailing

    def save(self, report_df: pd.DataFrame, model_id: str, date: str | date, overwrite: bool = False) -> str:
        path = self.paths.post_trade_result(model_id, date)
        if path.is_file() and not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
        StoragePaths.ensure_parent(path)
        report_df.to_parquet(path, index=False)
        return str(path)
