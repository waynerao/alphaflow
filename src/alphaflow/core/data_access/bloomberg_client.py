from datetime import date

import pandas as pd

from alphaflow.core.config.system_config import BloombergConfig
from alphaflow.core.data_access.exceptions import DataNotFoundError, DataSourceError
from alphaflow.core.integrations.desktool_adapter import DeskToolProtocol, get_desktool

RIC_COLUMN = "RIC"
BBL_COLUMN = "BBL"


class BBGClient:
    """Bloomberg reads via apcr_desktool.

    BBL exists only inside this class: RICs are mapped to BBLs on the way in and the results
    are mapped straight back, so no BBL column ever reaches a caller (spec section 4.1).
    """

    def __init__(self, config: BloombergConfig, desktool: DeskToolProtocol | None = None) -> None:
        self.config = config
        self._dt = desktool if desktool is not None else get_desktool()

    def bdh(self, rics: list[str], fields: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        ric_by_bbl = self._map_to_bbl(rics)
        try:
            df = self._dt.bdh(bbls=list(ric_by_bbl), fields=fields, start_date=start_date, end_date=end_date)
        except Exception as exc:
            raise DataSourceError(f"Bloomberg bdh failed: {exc}") from exc
        return self._to_ric(df, ric_by_bbl, "bdh")

    def bdp(self, rics: list[str], fields: list[str]) -> pd.DataFrame:
        ric_by_bbl = self._map_to_bbl(rics)
        try:
            df = self._dt.bdp(bbls=list(ric_by_bbl), fields=fields)
        except Exception as exc:
            raise DataSourceError(f"Bloomberg bdp failed: {exc}") from exc
        return self._to_ric(df, ric_by_bbl, "bdp")

    def _map_to_bbl(self, rics: list[str]) -> dict[str, str]:
        if not rics:
            raise ValueError("rics must not be empty")
        bbl_by_ric = self._dt.ric_to_bbl(rics)
        unmapped = [r for r in rics if not bbl_by_ric.get(r)]
        if unmapped:
            raise DataSourceError(f"no Bloomberg ticker mapping for {unmapped}")
        return {bbl_by_ric[ric]: ric for ric in rics}

    @classmethod
    def _to_ric(cls, df: pd.DataFrame, ric_by_bbl: dict[str, str], call: str) -> pd.DataFrame:
        if df is None or df.empty:
            raise DataNotFoundError(f"Bloomberg {call} returned no rows")
        out = df.copy()
        # Ticker may arrive as the index or as a column, under any of a few names
        if out.index.name in ("ticker", "bbl", BBL_COLUMN):
            out = out.reset_index().rename(columns={out.index.name: BBL_COLUMN})
        else:
            source = next((c for c in out.columns if c.lower() in ("bbl", "ticker", "security")), None)
            if source is None:
                raise DataSourceError(f"Bloomberg {call} returned no ticker column; got {list(out.columns)}")
            out = out.rename(columns={source: BBL_COLUMN})
        out[RIC_COLUMN] = out[BBL_COLUMN].map(ric_by_bbl)
        if out[RIC_COLUMN].isna().any():
            unknown = sorted(out.loc[out[RIC_COLUMN].isna(), BBL_COLUMN].unique())
            raise DataSourceError(f"Bloomberg {call} returned unrequested tickers: {unknown}")
        return out.drop(columns=[BBL_COLUMN]).reset_index(drop=True)
