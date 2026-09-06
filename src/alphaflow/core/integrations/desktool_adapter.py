"""Single integration seam for apcr_desktool.

apcr_desktool is the on-prem package that owns the kdb+ session, the Bloomberg session and
the BBL<->RIC symbology. AlphaFlow talks to it only through DeskToolProtocol, so:

  - nothing else in the codebase imports apcr_desktool;
  - the whole system is unit-testable off-prem against StubDeskTool;
  - reconciling with the real API is an edit to RealDeskTool alone.

Selection is by $ALPHAFLOW_DESKTOOL:  "real" | "stub" | "auto" (default).
"auto" uses the real package when importable and falls back to the stub otherwise.
"""

import os
from datetime import date
from typing import Any, Protocol, runtime_checkable

import pandas as pd

DESKTOOL_ENV_VAR = "ALPHAFLOW_DESKTOOL"
MODE_REAL, MODE_STUB, MODE_AUTO = "real", "stub", "auto"


@runtime_checkable
class DeskToolProtocol(Protocol):
    """The narrow surface AlphaFlow needs. Every method returns RIC-keyed data except the
    two symbology helpers, which are the only place BBL is allowed to exist."""

    def authenticate(self) -> None: ...

    def ric_to_bbl(self, rics: list[str]) -> dict[str, str]: ...

    def bbl_to_ric(self, bbls: list[str]) -> dict[str, str]: ...

    def query_kdb(self, table: str, rics: list[str], fields: list[str], start_date: date, end_date: date) -> pd.DataFrame: ...

    def query_kdb_latest(self, table: str, rics: list[str], fields: list[str]) -> pd.DataFrame: ...

    def bdh(self, bbls: list[str], fields: list[str], start_date: date, end_date: date) -> pd.DataFrame: ...

    def bdp(self, bbls: list[str], fields: list[str]) -> pd.DataFrame: ...


class DeskToolUnavailableError(RuntimeError):
    """Raised when real desktool access is required but the package is not importable."""


class StubDeskTool:
    """Off-prem placeholder. Symbology is a deterministic string transform; every data call
    raises, so a test that forgets to mock a data path fails loudly instead of inventing
    numbers. Unit tests inject their own doubles rather than relying on this."""

    def authenticate(self) -> None:
        return None

    def ric_to_bbl(self, rics: list[str]) -> dict[str, str]:
        return {ric: _ric_to_bbl_guess(ric) for ric in rics}

    def bbl_to_ric(self, bbls: list[str]) -> dict[str, str]:
        return {bbl: _bbl_to_ric_guess(bbl) for bbl in bbls}

    def query_kdb(self, table: str, rics: list[str], fields: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        raise DeskToolUnavailableError(f"kdb query for table {table!r} needs the real apcr_desktool; set {DESKTOOL_ENV_VAR}=real on-prem")

    def query_kdb_latest(self, table: str, rics: list[str], fields: list[str]) -> pd.DataFrame:
        raise DeskToolUnavailableError(f"kdb query for table {table!r} needs the real apcr_desktool; set {DESKTOOL_ENV_VAR}=real on-prem")

    def bdh(self, bbls: list[str], fields: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        raise DeskToolUnavailableError(f"Bloomberg history needs the real apcr_desktool; set {DESKTOOL_ENV_VAR}=real on-prem")

    def bdp(self, bbls: list[str], fields: list[str]) -> pd.DataFrame:
        raise DeskToolUnavailableError(f"Bloomberg reference needs the real apcr_desktool; set {DESKTOOL_ENV_VAR}=real on-prem")


# Placeholder symbology. HK RICs are zero-padded 4-digit codes; "0700.HK" <-> "700 HK Equity".
_SUFFIX_TO_BBG = {"HK": "HK", "SS": "CH", "SZ": "CH", "SI": "SP", "T": "JT", "KS": "KS", "AX": "AU", "TW": "TT"}
_BBG_TO_SUFFIX = {"HK": "HK", "SP": "SI", "JT": "T", "KS": "KS", "AU": "AX", "TT": "TW"}


def _ric_to_bbl_guess(ric: str) -> str:
    code, _, suffix = ric.rpartition(".")
    return f"{code.lstrip('0') or code} {_SUFFIX_TO_BBG.get(suffix, suffix)} Equity"


def _bbl_to_ric_guess(bbl: str) -> str:
    parts = bbl.split()
    if len(parts) < 2:
        return bbl
    code, exchange = parts[0], parts[1]
    suffix = _BBG_TO_SUFFIX.get(exchange, exchange)
    return f"{code.zfill(4)}.{suffix}" if suffix == "HK" else f"{code}.{suffix}"


class RealDeskTool:
    """Thin translation onto the installed apcr_desktool.

    TODO(on-prem): the method names below are AlphaFlow's assumed contract. Reconcile them
    once against the real package - this class is the only file that should need editing.
    """

    def __init__(self, module: Any) -> None:
        self._dt = module

    def authenticate(self) -> None:
        self._dt.authenticate()

    def ric_to_bbl(self, rics: list[str]) -> dict[str, str]:
        return self._dt.ric_to_bbl(rics)

    def bbl_to_ric(self, bbls: list[str]) -> dict[str, str]:
        return self._dt.bbl_to_ric(bbls)

    def query_kdb(self, table: str, rics: list[str], fields: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        return self._dt.query_kdb(table=table, rics=rics, fields=fields, start_date=start_date, end_date=end_date)

    def query_kdb_latest(self, table: str, rics: list[str], fields: list[str]) -> pd.DataFrame:
        return self._dt.query_kdb_latest(table=table, rics=rics, fields=fields)

    def bdh(self, bbls: list[str], fields: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        return self._dt.bdh(tickers=bbls, flds=fields, start_date=start_date, end_date=end_date)

    def bdp(self, bbls: list[str], fields: list[str]) -> pd.DataFrame:
        return self._dt.bdp(tickers=bbls, flds=fields)


_cached: DeskToolProtocol | None = None


def _import_desktool() -> Any | None:
    try:
        import apcr_desktool
        return apcr_desktool
    except ImportError:
        return None


def get_desktool(mode: str | None = None, refresh: bool = False) -> DeskToolProtocol:
    """Process-wide desktool handle. refresh=True forces re-resolution (used by tests)."""
    global _cached
    if _cached is not None and not refresh:
        return _cached
    resolved_mode = (mode or os.environ.get(DESKTOOL_ENV_VAR, MODE_AUTO)).lower()
    if resolved_mode not in (MODE_REAL, MODE_STUB, MODE_AUTO):
        raise ValueError(f"{DESKTOOL_ENV_VAR} must be one of real|stub|auto, got {resolved_mode!r}")
    module = None if resolved_mode == MODE_STUB else _import_desktool()
    if resolved_mode == MODE_REAL and module is None:
        raise DeskToolUnavailableError(f"{DESKTOOL_ENV_VAR}=real but apcr_desktool is not importable; "
                                       "install it with 'uv pip install -e ../apcr_desktool'")
    _cached = RealDeskTool(module) if module is not None else StubDeskTool()
    return _cached


def reset_desktool() -> None:
    global _cached
    _cached = None
