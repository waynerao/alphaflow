from datetime import date

import pandas as pd

from alphaflow.core.config.system_config import StorageConfig
from alphaflow.core.utils.storage import StoragePaths
from alphaflow.strategy_backtester.engine.backtest_engine import MEASURE_ORDER
from alphaflow.strategy_backtester.risk.barra_exposure import BARRA_PREFIX

MEASURE_COLUMN = "measure"


class ReportGenerator:
    """Builds the measures x rtn_alpha_id table (spec section 7.3) and persists it."""

    def __init__(self, storage_config: StorageConfig) -> None:
        self.paths = StoragePaths(storage_config)

    def assemble(self, results: dict[str, dict[str, float]]) -> pd.DataFrame:
        """results maps rtn_alpha_id -> {measure: value}. Rows are measures, one column per
        return definition tested, so two rtn alphas sit side by side for comparison."""
        if not results:
            return pd.DataFrame(columns=[MEASURE_COLUMN])
        table = pd.DataFrame(results)
        table = table.reindex(self._ordered_measures(table.index))
        return table.rename_axis(MEASURE_COLUMN).reset_index()

    @classmethod
    def _ordered_measures(cls, measures: pd.Index) -> list[str]:
        """Fixed measures first in their canonical order, then Barra factors alphabetically,
        then anything else - so the table reads the same way every run."""
        present = list(measures)
        fixed = [m for m in MEASURE_ORDER if m in present]
        barra = sorted(m for m in present if m.startswith(BARRA_PREFIX))
        remainder = [m for m in present if m not in fixed and m not in barra]
        return fixed + barra + remainder

    def save(self, result_df: pd.DataFrame, market: str, alpha_id: str, start_date: str | date,
             end_date: str | date, overwrite: bool = False) -> str:
        """Backtest results are always saved. An existing file is a signal that this exact
        window was already run, so overwriting takes an explicit flag."""
        path = self.paths.backtest_result(market, alpha_id, start_date, end_date)
        if path.is_file() and not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
        StoragePaths.ensure_parent(path)
        result_df.to_parquet(path, index=False)
        return str(path)
