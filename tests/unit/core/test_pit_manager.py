from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from alphaflow.core.data_access.bloomberg_client import BBGClient
from alphaflow.core.data_access.exceptions import DataNotFoundError, DataSourceError, PITViolationError
from alphaflow.core.data_access.kdb_client import KDBClient
from alphaflow.core.data_access.pit_manager import PITDataManager, PITMode

AS_OF = date(2024, 1, 15)


def kdb_frame(stamps, rics=("0700.HK", "0941.HK")):
    return pd.DataFrame([{"RIC": ric, "timestamp": stamp, "close": 100.0} for stamp in stamps for ric in rics])


class TestConstruction:
    def test_research_requires_an_as_of_date(self, system_config):
        with pytest.raises(ValueError, match="requires an explicit as_of_date"):
            PITDataManager(system_config, mode=PITMode.RESEARCH)

    def test_unknown_mode_rejected(self, system_config):
        with pytest.raises(ValueError, match="mode must be"):
            PITDataManager(system_config, mode="sideways")

    def test_production_defaults_to_today(self, system_config, fake_desktool):
        manager = PITDataManager.for_production(system_config, desktool=fake_desktool())
        assert manager.as_of_date == date.today() and not manager.is_research


class TestPITEnforcement:
    def test_research_rejects_rows_stamped_after_as_of(self, system_config, fake_desktool):
        future = kdb_frame([datetime(2024, 1, 16, 10, 0)])
        manager = PITDataManager.for_research(system_config, AS_OF, desktool=fake_desktool(kdb_result=future))
        with pytest.raises(PITViolationError, match="stamped after as_of_date"):
            manager.get_minute_bar(["0700.HK"], "HK", AS_OF - timedelta(days=5))

    def test_research_accepts_rows_up_to_end_of_the_as_of_day(self, system_config, fake_desktool):
        same_day = kdb_frame([datetime(2024, 1, 15, 23, 59)])
        manager = PITDataManager.for_research(system_config, AS_OF, desktool=fake_desktool(kdb_result=same_day))
        assert len(manager.get_minute_bar(["0700.HK"], "HK", AS_OF)) == 2

    def test_production_does_not_filter(self, system_config, fake_desktool):
        future = kdb_frame([datetime(2099, 1, 1, 10, 0)])
        manager = PITDataManager.for_production(system_config, desktool=fake_desktool(kdb_result=future))
        assert len(manager.get_minute_bar(["0700.HK"], "HK", AS_OF)) == 2

    def test_research_uses_a_bounded_query_production_uses_latest(self, system_config, fake_desktool):
        frame = kdb_frame([datetime(2024, 1, 10, 10, 0)])
        research_dt, production_dt = fake_desktool(kdb_result=frame), fake_desktool(kdb_result=frame)
        PITDataManager.for_research(system_config, AS_OF, desktool=research_dt).get_minute_bar(["0700.HK"], "HK", AS_OF)
        PITDataManager.for_production(system_config, desktool=production_dt).get_minute_bar(["0700.HK"], "HK", AS_OF)
        assert research_dt.calls[0][0] == "query_kdb" and research_dt.calls[0][1]["end_date"] == AS_OF
        assert production_dt.calls[0][0] == "query_kdb_latest"

    def test_a_frame_without_a_timestamp_column_passes_through(self, system_config, fake_desktool):
        frame = pd.DataFrame({"RIC": ["0700.HK"], "close": [100.0]})
        manager = PITDataManager.for_research(system_config, AS_OF, desktool=fake_desktool(kdb_result=frame))
        assert len(manager.get_minute_bar(["0700.HK"], "HK", AS_OF)) == 1


class TestDataTypeAvailability:
    def test_unavailable_data_type_rejected_for_the_market(self, system_config, fake_desktool):
        manager = PITDataManager.for_production(system_config, desktool=fake_desktool(kdb_result=kdb_frame([datetime(2024, 1, 1)])))
        with pytest.raises(KeyError, match="not available for KR"):
            manager.get_indic(["005930.KS"], "KR", AS_OF, "open")

    def test_empty_result_raises_not_found(self, system_config, fake_desktool):
        manager = PITDataManager.for_production(system_config, desktool=fake_desktool(kdb_result=pd.DataFrame()))
        with pytest.raises(DataNotFoundError):
            manager.get_minute_bar(["0700.HK"], "HK", AS_OF)


class TestRICContract:
    def test_a_sym_column_is_renamed_to_ric(self, system_config, fake_desktool):
        frame = pd.DataFrame({"sym": ["0700.HK"], "close": [100.0]})
        manager = PITDataManager.for_production(system_config, desktool=fake_desktool(kdb_result=frame))
        assert "RIC" in manager.get_minute_bar(["0700.HK"], "HK", AS_OF).columns

    def test_no_symbol_column_at_all_is_an_error(self, system_config, fake_desktool):
        client = KDBClient(system_config.kdb, desktool=fake_desktool(kdb_result=pd.DataFrame({"close": [1.0]})))
        with pytest.raises(DataSourceError, match="no RIC column"):
            client.query_latest("t", ["0700.HK"], ["close"])

    def test_empty_ric_list_rejected(self, system_config, fake_desktool):
        client = KDBClient(system_config.kdb, desktool=fake_desktool())
        with pytest.raises(ValueError, match="must not be empty"):
            client.query_latest("t", [], ["close"])


class TestBloombergNeverLeaksBBL:
    def test_bdp_returns_ric_and_no_bbl(self, system_config, fake_desktool):
        result = pd.DataFrame({"ticker": ["700 HK Equity", "941 HK Equity"], "PX_LAST": [1.0, 2.0]})
        client = BBGClient(system_config.bloomberg, desktool=fake_desktool(bdp_result=result))
        out = client.bdp(["0700.HK", "0941.HK"], ["PX_LAST"])
        assert list(out["RIC"]) == ["0700.HK", "0941.HK"]
        assert not any(c.upper() in ("BBL", "TICKER") for c in out.columns)

    def test_desktool_receives_bbls_not_rics(self, system_config, fake_desktool):
        desktool = fake_desktool(bdp_result=pd.DataFrame({"ticker": ["700 HK Equity"], "PX_LAST": [1.0]}))
        BBGClient(system_config.bloomberg, desktool=desktool).bdp(["0700.HK"], ["PX_LAST"])
        assert desktool.calls[0][1]["bbls"] == ["700 HK Equity"]

    def test_an_unrequested_ticker_in_the_response_is_an_error(self, system_config, fake_desktool):
        result = pd.DataFrame({"ticker": ["9999 HK Equity"], "PX_LAST": [1.0]})
        client = BBGClient(system_config.bloomberg, desktool=fake_desktool(bdp_result=result))
        with pytest.raises(DataSourceError, match="unrequested tickers"):
            client.bdp(["0700.HK"], ["PX_LAST"])

    def test_ticker_supplied_as_the_index_is_handled(self, system_config, fake_desktool):
        result = pd.DataFrame({"PX_LAST": [1.0]}, index=pd.Index(["700 HK Equity"], name="ticker"))
        client = BBGClient(system_config.bloomberg, desktool=fake_desktool(bdp_result=result))
        assert list(client.bdp(["0700.HK"], ["PX_LAST"])["RIC"]) == ["0700.HK"]


class TestRiskFactors:
    def test_factors_are_filtered_to_the_requested_model(self, system_config, fake_desktool, risk_frame_factory):
        frame = pd.concat([risk_frame_factory(model="ASE2S"), risk_frame_factory(model="OTHER")], ignore_index=True)
        manager = PITDataManager.for_production(system_config, desktool=fake_desktool(kdb_result=frame))
        out = manager.get_risk_factors(["0700.HK"], "HK", AS_OF, model="ASE2S")
        assert set(out["model"]) == {"ASE2S"}
