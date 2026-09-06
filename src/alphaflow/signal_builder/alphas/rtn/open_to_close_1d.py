from datetime import date

import pandas as pd

from alphaflow.core.alpha_base.rtn_alpha import RtnAlpha

RIC_COLUMN = "RIC"
TIMESTAMP_COLUMN = "timestamp"
OPEN_COLUMN = "open"
CLOSE_COLUMN = "close"
VOLUME_COLUMN = "volume"


class OpenToClose1D(RtnAlpha):
    """Next-day AM open to PM close return.

    raw    = close(t + horizon, exit session) / open(t + horizon, entry session) - 1
    hedged = raw - the cap-neutral cross-sectional mean, a stand-in for the index return
             when no index series is loaded.

    When return_type requests both, the hedged series is returned - it is the one the
    backtester and post_trade_analyzer should be scoring against.
    """

    def compute(self, market: str, as_of_date: date) -> pd.Series:
        prices = self._session_prices()    # MultiIndex (day, RIC), columns entry/exit
        days = sorted(prices.index.get_level_values(0).unique())
        target = prices.xs(self._target_day(days, as_of_date), level=0)
        entry, exit_price = target["entry"], target["exit"]
        raw = (exit_price / entry.where(entry > 0) - 1.0).dropna()
        raw.name = self.alpha_id
        return self._hedge(raw) if "hedged" in self.return_type else raw

    def _hedge(self, raw: pd.Series) -> pd.Series:
        """Equal-weighted cross-sectional demeaning. Swap in a real index return here once
        an index series is part of required_inputs."""
        hedged = raw - raw.mean()
        hedged.name = raw.name
        return hedged

    def _target_day(self, days: list, as_of_date: date):
        """The day whose prices realise the return signalled on as_of_date: horizon steps
        forward, counted along the days actually present in the data."""
        available = list(days)
        if as_of_date not in available:
            raise ValueError(f"{self.alpha_id}: as_of_date {as_of_date} absent from working data ({available[:1]}..{available[-1:]})")
        index = available.index(as_of_date) + self.horizon
        if index >= len(available):
            raise ValueError(f"{self.alpha_id}: horizon {self.horizon} runs past the end of the working data at {as_of_date}")
        return available[index]

    def _session_prices(self) -> pd.DataFrame:
        """Per (day, RIC): the entry-session price and the exit-session price, chosen from
        the config's PricePoints. Indexed by day, then RIC."""
        df = self.working_data
        missing = [c for c in (RIC_COLUMN, TIMESTAMP_COLUMN, OPEN_COLUMN, CLOSE_COLUMN) if c not in df.columns]
        if missing:
            raise ValueError(f"{self.alpha_id}: working data missing columns {missing}")
        frame = df.copy()
        frame["_stamp"] = pd.to_datetime(frame[TIMESTAMP_COLUMN])
        frame["_day"] = frame["_stamp"].dt.date
        entry = self._price_for(frame, self.config.entry).rename("entry")
        exit_price = self._price_for(frame, self.config.exit).rename("exit")
        return pd.concat([entry, exit_price], axis=1).dropna()

    def _price_for(self, frame: pd.DataFrame, point) -> pd.Series:
        """Reduce minute bars to one price per (day, RIC) for a single PricePoint."""
        subset = self._session_slice(frame, point.session)
        grouped = subset.sort_values("_stamp").groupby(["_day", RIC_COLUMN])
        if point.price_type == "open":
            return grouped[OPEN_COLUMN].first()
        if point.price_type == "close":
            return grouped[CLOSE_COLUMN].last()
        window = subset[(subset["_time"] >= point.vwap_start) & (subset["_time"] <= point.vwap_end)]
        if VOLUME_COLUMN not in window.columns:
            raise ValueError(f"{self.alpha_id}: vwap requires a '{VOLUME_COLUMN}' column")
        turnover = (window[CLOSE_COLUMN] * window[VOLUME_COLUMN]).groupby([window["_day"], window[RIC_COLUMN]]).sum()
        volume = window[VOLUME_COLUMN].groupby([window["_day"], window[RIC_COLUMN]]).sum()
        return turnover / volume.where(volume > 0)

    def _session_slice(self, frame: pd.DataFrame, session: str | None) -> pd.DataFrame:
        """Restrict to the am or pm session using the market's own session table. With no
        session set - or no market config to consult - the whole day is in scope."""
        out = frame.copy()
        out["_time"] = out["_stamp"].dt.strftime("%H:%M")
        if session is None or self.market_config is None:
            return out
        market = str(self.market_scope[0]) if len(self.market_scope) == 1 else None
        sessions = self.market_config[market].sessions if market in (self.market_config or {}) else None
        if not sessions or len(sessions) < 2:
            return out
        window = sessions[0] if session == "am" else sessions[-1]
        return out[(out["_time"] >= window.start) & (out["_time"] <= window.end)]
