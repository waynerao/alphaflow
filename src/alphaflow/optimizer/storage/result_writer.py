import pandas as pd

from alphaflow.core.config.system_config import StorageConfig
from alphaflow.core.utils.storage import StoragePaths

ROW_ID_COLUMN = "row_id"
PORTFOLIO_ROW = "PORTFOLIO"
NOT_APPLICABLE = float("nan")
LEADING_COLUMNS = ["weight", "sharpe", "std", "mean_return", "avg_corr"]


class ResultWriter:
    """One wide audit table per model: a row per alpha plus a PORTFOLIO summary row."""

    def __init__(self, storage_config: StorageConfig) -> None:
        self.paths = StoragePaths(storage_config)

    def build_table(self, weights: pd.Series, standalone_metrics: dict[str, dict[str, float]],
                    avg_correlations: dict[str, float], standalone_barra: dict[str, dict[str, float]],
                    combined_stats: dict[str, float], combined_barra: dict[str, float]) -> pd.DataFrame:
        rows = {}
        for alpha_id in weights.index:
            metrics = standalone_metrics.get(alpha_id, {})
            rows[alpha_id] = {
                "weight": float(weights[alpha_id]),
                "sharpe": metrics.get("standalone_sharpe"),
                "std": metrics.get("standalone_std"),
                "mean_return": metrics.get("standalone_mean_return"),
                "avg_corr": avg_correlations.get(alpha_id),
                **standalone_barra.get(alpha_id, {}),
            }
        # avg_corr is meaningless for the portfolio as a whole - it has nothing to correlate against
        rows[PORTFOLIO_ROW] = {
            "weight": float(weights.sum()),
            "sharpe": combined_stats.get("standalone_sharpe"),
            "std": combined_stats.get("standalone_std"),
            "mean_return": combined_stats.get("standalone_mean_return"),
            "avg_corr": NOT_APPLICABLE,
            **combined_barra,
        }
        table = pd.DataFrame.from_dict(rows, orient="index")
        return table[self._ordered_columns(table.columns)].rename_axis(ROW_ID_COLUMN).reset_index()

    @classmethod
    def _ordered_columns(cls, columns: pd.Index) -> list[str]:
        leading = [c for c in LEADING_COLUMNS if c in columns]
        return leading + sorted(c for c in columns if c not in leading)

    def save(self, result_df: pd.DataFrame, market: str, model_id: str, overwrite: bool = False) -> str:
        path = self.paths.optimizer_result(market, model_id)
        if path.is_file() and not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
        StoragePaths.ensure_parent(path)
        result_df.to_parquet(path, index=False)
        return str(path)
