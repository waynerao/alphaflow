from datetime import date, timedelta

import pandas as pd
import pytest

from alphaflow.core.utils.storage import StoragePaths

MARKET = "HK"
AS_OF = date(2024, 1, 31)
MOMENTUM, IMBALANCE, RETURN = "momentum_price_21d", "order_imbalance_1430", "open_to_close_1d"


@pytest.fixture
def raw_data_window(system_config, minute_bar_frame, depth_frame):
    """The CSVs data_dumper would have written, wide enough for a 21d lookback and a 1d
    forward horizon around a run of scoring dates."""
    def build(scoring_days):
        paths = StoragePaths(system_config.storage)
        earliest = min(scoring_days) - timedelta(days=35)
        latest = max(scoring_days) + timedelta(days=9)
        day = earliest
        while day <= latest:
            path = StoragePaths.ensure_parent(paths.raw_data(MARKET, "minute_bar", day))
            minute_bar_frame([day]).to_csv(path, index=False)
            depth_path = StoragePaths.ensure_parent(paths.raw_data(MARKET, "depth_intraday", day))
            depth_frame(day).to_csv(depth_path, index=False)
            day += timedelta(days=1)
        return scoring_days
    return build


@pytest.fixture
def scoring_days():
    return [AS_OF - timedelta(days=offset) for offset in range(11, -1, -1)]


@pytest.fixture
def date_strings():
    def build(days):
        return [d.strftime("%Y%m%d") for d in days]
    return build


@pytest.fixture
def wide_risk_pit(system_config, risk_frame_factory, scoring_days, fake_desktool):
    """A research PITDataManager whose risk frame spans every (RIC, date), so the pooled
    attribution regression has far more observations than factors."""
    from alphaflow.core.data_access.pit_manager import PITDataManager

    per_day = []
    for index, day in enumerate(scoring_days):
        frame = risk_frame_factory(seed=100 + index)
        frame["date"] = day
        per_day.append(frame)
    risk = pd.concat(per_day, ignore_index=True)
    manager = PITDataManager.for_research(system_config, max(scoring_days), desktool=fake_desktool())
    manager.get_risk_factors = lambda rics, market, start_date, model="ASE2S": risk[risk["RIC"].isin(rics)]
    return manager
