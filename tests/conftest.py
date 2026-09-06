from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from alphaflow.core.alpha_base.registry import AlphaRegistry
from alphaflow.core.config.loader import (
    load_optimizer_config,
    load_production_config,
    load_registry_config,
    load_system_config,
)
from alphaflow.core.config.paths import SHARED_DRIVE_ENV_VAR
from alphaflow.core.data_access.pit_manager import PITDataManager
from alphaflow.core.integrations.desktool_adapter import reset_desktool

RICS = ["0700.HK", "0941.HK", "0005.HK", "1299.HK", "0388.HK"]
MARKET = "HK"
BARRA_FACTORS = ["mkt", "size", "value", "momentum", "ind_tech", "ind_financials"]


@pytest.fixture(autouse=True)
def isolated_shared_drive(tmp_path, monkeypatch):
    """Point every StoragePaths at a per-test temp directory. Autouse, so no test can
    accidentally write into the repo's data/ directory."""
    root = tmp_path / "shared"
    root.mkdir()
    monkeypatch.setenv(SHARED_DRIVE_ENV_VAR, str(root))
    reset_desktool()
    yield root
    reset_desktool()


@pytest.fixture
def system_config():
    return load_system_config()


@pytest.fixture
def production_config():
    return load_production_config()


@pytest.fixture
def optimizer_config():
    return load_optimizer_config()


@pytest.fixture
def registry_config():
    return load_registry_config()


@pytest.fixture
def registry(system_config):
    return AlphaRegistry.from_paths(production_path="configs/production.toml", system_config=system_config)


@pytest.fixture
def trading_dates():
    return [date(2024, 1, 1) + timedelta(days=offset) for offset in range(20)]


@pytest.fixture
def score_frame_factory():
    """Builds a stored-score frame (date, RIC, raw_signal, score) with an optional planted
    correlation to a companion return frame."""
    def build(dates, rics=RICS, seed=0, scale=1.0):
        rng = np.random.default_rng(seed)
        rows = []
        for day in dates:
            for ric in rics:
                value = float(rng.normal(0, scale))
                rows.append({"date": day, "RIC": ric, "raw_signal": value, "score": value})
        return pd.DataFrame(rows)
    return build


@pytest.fixture
def correlated_frames(trading_dates):
    """(alpha_scores, return_scores) where forward return = beta * score + noise."""
    def build(beta=0.5, noise=0.5, seed=1, rics=RICS, dates=None):
        rng = np.random.default_rng(seed)
        days = dates if dates is not None else trading_dates
        alpha_rows, return_rows = [], []
        for day in days:
            scores = rng.normal(size=len(rics))
            returns = beta * scores + rng.normal(scale=noise, size=len(rics))
            for ric, s, r in zip(rics, scores, returns):
                alpha_rows.append({"date": day, "RIC": ric, "raw_signal": s, "score": s})
                return_rows.append({"date": day, "RIC": ric, "raw_signal": r, "score": r})
        return pd.DataFrame(alpha_rows), pd.DataFrame(return_rows)
    return build


@pytest.fixture
def risk_frame_factory():
    """A Barra-style exposure frame. Factor columns are arbitrary on purpose - nothing in
    the codebase may depend on a fixed factor list."""
    def build(rics=RICS, factors=BARRA_FACTORS, seed=2, model="ASE2S"):
        rng = np.random.default_rng(seed)
        data = {"RIC": rics, "model": [model] * len(rics)}
        data.update({factor: rng.normal(size=len(rics)) for factor in factors})
        return pd.DataFrame(data)
    return build


@pytest.fixture
def minute_bar_frame():
    """Minute bars spanning both HK sessions, for several days."""
    def build(days, rics=RICS, seed=3):
        rows = []
        for day in days:
            # Seed off the date so a per-day call and a multi-day call agree, while prices
            # still move between days - a flat series would make every momentum exactly zero
            rng = np.random.default_rng(seed + day.toordinal())
            for index, ric in enumerate(rics):
                base = 100.0 * (1.0 + 0.01 * index) * (1.0 + 0.0004 * day.toordinal() % 0.3)
                for hour, minute in [(9, 30), (11, 59), (13, 0), (15, 59)]:
                    price = base * (1 + rng.normal(0, 0.005))
                    rows.append({"RIC": ric, "timestamp": datetime(day.year, day.month, day.day, hour, minute),
                                 "open": price, "high": price * 1.01, "low": price * 0.99,
                                 "close": price, "volume": 1000 + index})
        return pd.DataFrame(rows)
    return build


@pytest.fixture
def depth_frame():
    """Five-level order book, bid-heavy for later RICs so imbalance is monotone in index."""
    def build(day, rics=RICS, levels=5, hour=14, minute=29):
        rows = []
        for index, ric in enumerate(rics):
            for level in range(1, levels + 1):
                rows.append({"RIC": ric, "timestamp": datetime(day.year, day.month, day.day, hour, minute),
                             "level": level, "bid_price": 100 - level * 0.1, "bid_size": 500 + index * 100,
                             "ask_price": 100 + level * 0.1, "ask_size": 400})
        return pd.DataFrame(rows)
    return build


class FakeDeskTool:
    """Records calls and returns whatever the test queued. Injected wherever a client would
    otherwise reach the real desktool."""

    def __init__(self, kdb_result=None, bdh_result=None, bdp_result=None):
        self.kdb_result = kdb_result
        self.bdh_result = bdh_result
        self.bdp_result = bdp_result
        self.calls = []

    def authenticate(self):
        self.calls.append(("authenticate", {}))

    def ric_to_bbl(self, rics):
        return {ric: f"{ric.split('.')[0].lstrip('0')} HK Equity" for ric in rics}

    def bbl_to_ric(self, bbls):
        return {bbl: f"{bbl.split()[0].zfill(4)}.HK" for bbl in bbls}

    def query_kdb(self, table, rics, fields, start_date, end_date):
        self.calls.append(("query_kdb", {"table": table, "rics": rics, "start_date": start_date, "end_date": end_date}))
        return self.kdb_result

    def query_kdb_latest(self, table, rics, fields):
        self.calls.append(("query_kdb_latest", {"table": table, "rics": rics}))
        return self.kdb_result

    def bdh(self, bbls, fields, start_date, end_date):
        self.calls.append(("bdh", {"bbls": bbls, "fields": fields}))
        return self.bdh_result

    def bdp(self, bbls, fields):
        self.calls.append(("bdp", {"bbls": bbls, "fields": fields}))
        return self.bdp_result


@pytest.fixture
def fake_desktool():
    return FakeDeskTool


@pytest.fixture
def research_pit(system_config, risk_frame_factory):
    """A research-mode PITDataManager whose risk-factor call is stubbed."""
    def build(as_of=date(2024, 1, 20), risk=None):
        manager = PITDataManager.for_research(system_config, as_of, desktool=FakeDeskTool())
        frame = risk if risk is not None else risk_frame_factory()
        manager.get_risk_factors = lambda rics, market, start_date, model="ASE2S": frame[frame["RIC"].isin(rics)]
        return manager
    return build
