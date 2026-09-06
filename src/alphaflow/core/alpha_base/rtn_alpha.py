from abc import abstractmethod
from datetime import date

import pandas as pd

from alphaflow.core.alpha_base.base_alpha import BaseAlpha
from alphaflow.core.alpha_base.market import DUAL_SESSION_MARKETS
from alphaflow.core.config.alpha_config.rtn_config import PricePoint, RtnAlphaConfig
from alphaflow.core.config.exceptions import ConfigValidationError
from alphaflow.core.config.paths import DEFAULT_UNIVERSE_DIR
from alphaflow.core.config.system_config import MarketConfig
from alphaflow.core.data_access.pit_manager import PITDataManager


class RtnAlpha(BaseAlpha):
    """Forward-return signal.

    Doubles as the realised-return source for strategy_backtester and post_trade_analyzer,
    and is the only alpha type carrying entry/exit timing.
    """

    EXPECTED_ALPHA_TYPE = "rtn"

    def __init__(self, config: RtnAlphaConfig, data: PITDataManager, universe_dir: str = DEFAULT_UNIVERSE_DIR,
                 market_config: dict[str, MarketConfig] | None = None) -> None:
        self.market_config = market_config
        super().__init__(config, data, universe_dir)

    @property
    def horizon(self) -> int:
        return self.config.horizon

    @property
    def return_type(self) -> list[str]:
        return self.config.return_type

    @abstractmethod
    def compute(self, market: str, as_of_date: date) -> pd.Series:
        """Return a pd.Series indexed by RIC whose values are forward returns."""

    def run(self, market: str, as_of_date: date) -> pd.Series:
        if self.config.is_intraday:
            self._validate_intraday_session(market)
        return super().run(market, as_of_date)

    def _validate_intraday_session(self, market: str) -> None:
        """horizon=0 means entry and exit fall on the same day, so the configured price
        points have to be reachable within that market's actual sessions - and the exit has
        to come after the entry. Skipped when no market config was supplied."""
        if self.market_config is None:
            return
        config = self.market_config.get(market)
        if config is None:
            raise ConfigValidationError(f"{self.alpha_id}: market {market!r} missing from system config")
        entry_time = self._resolve_time(self.config.entry, config, market, "entry")
        exit_time = self._resolve_time(self.config.exit, config, market, "exit")
        if entry_time >= exit_time:
            raise ConfigValidationError(f"{self.alpha_id}: intraday entry {entry_time} does not precede exit {exit_time} for {market}")

    def _resolve_time(self, point: PricePoint, config: MarketConfig, market: str, label: str) -> str:
        """Map a PricePoint onto a concrete HH:MM using the market's session table."""
        sessions = config.sessions
        is_dual = market in DUAL_SESSION_MARKETS
        if point.session is not None:
            if not is_dual:
                raise ConfigValidationError(f"{self.alpha_id}: {label}.session={point.session!r} but {market} is a single-session market")
            if len(sessions) < 2:
                raise ConfigValidationError(f"{self.alpha_id}: {market} is configured with {len(sessions)} session(s); "
                                            f"{label}.session={point.session!r} is unreachable")
        if point.price_type == "vwap":
            window_start, window_end = point.vwap_start, point.vwap_end
            if not self._within_sessions(window_start, sessions) or not self._within_sessions(window_end, sessions):
                raise ConfigValidationError(f"{self.alpha_id}: {label} vwap window {window_start}-{window_end} falls outside {market} sessions")
            return window_start if label == "entry" else window_end
        session = sessions[0] if point.session in (None, "am") else sessions[-1]
        return session.start if point.price_type == "open" else session.end

    @classmethod
    def _within_sessions(cls, time_str: str, sessions: list) -> bool:
        return any(s.start <= time_str <= s.end for s in sessions)
