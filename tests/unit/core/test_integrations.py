import logging

import pytest

from alphaflow.core.integrations.desktool_adapter import (
    DESKTOOL_ENV_VAR,
    DeskToolProtocol,
    DeskToolUnavailableError,
    RealDeskTool,
    StubDeskTool,
    get_desktool,
    reset_desktool,
)
from alphaflow.core.integrations.logging_setup import setup_logger


class TestAdapterSelection:
    def test_stub_is_used_when_desktool_is_absent(self):
        assert isinstance(get_desktool(mode="stub", refresh=True), StubDeskTool)

    def test_real_mode_fails_loudly_when_the_package_is_missing(self):
        with pytest.raises(DeskToolUnavailableError, match="not importable"):
            get_desktool(mode="real", refresh=True)

    def test_auto_falls_back_to_the_stub(self):
        assert isinstance(get_desktool(mode="auto", refresh=True), StubDeskTool)

    def test_env_var_selects_the_mode(self, monkeypatch):
        monkeypatch.setenv(DESKTOOL_ENV_VAR, "stub")
        assert isinstance(get_desktool(refresh=True), StubDeskTool)

    def test_an_invalid_mode_is_rejected(self):
        with pytest.raises(ValueError, match="must be one of real|stub|auto"):
            get_desktool(mode="sideways", refresh=True)

    def test_the_handle_is_cached_until_refreshed(self):
        first = get_desktool(mode="stub", refresh=True)
        assert get_desktool() is first
        reset_desktool()
        assert get_desktool(mode="stub") is not first

    def test_both_implementations_satisfy_the_protocol(self):
        assert isinstance(StubDeskTool(), DeskToolProtocol)
        assert isinstance(RealDeskTool(object()), DeskToolProtocol)


class TestStubBehaviour:
    def test_symbology_round_trips_for_hk(self):
        stub = StubDeskTool()
        bbl = stub.ric_to_bbl(["0700.HK"])["0700.HK"]
        assert bbl == "700 HK Equity" and stub.bbl_to_ric([bbl])[bbl] == "0700.HK"

    @pytest.mark.parametrize("call,args", [
        ("query_kdb", ("t", ["0700.HK"], ["close"], None, None)),
        ("query_kdb_latest", ("t", ["0700.HK"], ["close"])),
        ("bdh", (["700 HK Equity"], ["PX_LAST"], None, None)),
        ("bdp", (["700 HK Equity"], ["PX_LAST"])),
    ])
    def test_every_data_call_raises_rather_than_inventing_numbers(self, call, args):
        with pytest.raises(DeskToolUnavailableError, match="real apcr_desktool"):
            getattr(StubDeskTool(), call)(*args)

    def test_authenticate_is_a_no_op(self):
        assert StubDeskTool().authenticate() is None


class TestRealAdapterDelegation:
    def test_calls_are_forwarded_to_the_wrapped_module(self, mocker):
        module = mocker.Mock()
        adapter = RealDeskTool(module)
        adapter.query_kdb("tbl", ["0700.HK"], ["close"], "s", "e")
        adapter.ric_to_bbl(["0700.HK"])
        module.query_kdb.assert_called_once_with(table="tbl", rics=["0700.HK"], fields=["close"], start_date="s", end_date="e")
        module.ric_to_bbl.assert_called_once_with(["0700.HK"])


class TestLogging:
    def test_logger_is_configured_once(self):
        logger = setup_logger("alphaflow.test.once")
        assert logger.handlers and len(setup_logger("alphaflow.test.once").handlers) == len(logger.handlers)

    def test_a_file_handler_is_added_when_a_path_is_given(self, tmp_path):
        logger = setup_logger("alphaflow.test.file", log_path=tmp_path / "logs")
        logger.info("hello")
        for handler in logger.handlers:
            handler.flush()
        assert (tmp_path / "logs" / "alphaflow.log").is_file()

    def test_level_is_applied(self):
        assert setup_logger("alphaflow.test.level", level=logging.WARNING).level == logging.WARNING
