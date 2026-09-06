import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TIME_PATTERN = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

# Placeholder kdb+ table names, one per data_type. Spec section 5.1 defers the real schema;
# override [kdb.tables] in system.toml once the actual kdb+ layout is known.
DEFAULT_KDB_TABLES = {
    "depth_open": "depth_snapshot", "depth_close": "depth_snapshot", "depth_intraday": "depth_snapshot",
    "indic_open": "indicative", "indic_close": "indicative", "minute_bar": "minute_bar",
    "ohlcv": "daily_bar", "tick": "trade", "risk_factors": "barra_exposure",
}

# Placeholder S3 key templates. Substituted with {market} and {date} (YYYYMMDD).
DEFAULT_S3_KEYS = {
    "intropic": "intropic/{market}/{date}.csv",
    "southbound": "southbound/{market}/{date}.csv",
}


def validate_hhmm(value: str) -> str:
    if not TIME_PATTERN.match(value):
        raise ValueError(f"expected HH:MM 24-hour time, got {value!r}")
    return value


class SessionConfig(BaseModel):
    start: str
    end: str

    _check_start = field_validator("start")(staticmethod(validate_hhmm))
    _check_end = field_validator("end")(staticmethod(validate_hhmm))

    @model_validator(mode="after")
    def validate_order(self) -> "SessionConfig":
        if self.start >= self.end:
            raise ValueError(f"session start {self.start} must precede end {self.end}")
        return self


class MarketConfig(BaseModel):
    available_data_types: list[str]
    sessions: list[SessionConfig] = Field(min_length=1)


class StorageConfig(BaseModel):
    shared_drive_path: str
    parquet_cold_path: str
    log_path: str


class KdbConfig(BaseModel):
    """host/port are retained for the on-prem desktool connection profile. AlphaFlow itself
    never opens a kdb+ socket - every read goes through the desktool adapter (decision D13)."""
    host: str
    port: int = Field(gt=0, lt=65536)
    tables: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_KDB_TABLES))

    def table_for(self, data_type: str) -> str:
        if data_type not in self.tables:
            raise KeyError(f"no kdb table mapped for data_type {data_type!r}; add it to [kdb.tables] in system.toml")
        return self.tables[data_type]


class BloombergConfig(BaseModel):
    timeout_seconds: int = Field(default=30, gt=0)


class S3Config(BaseModel):
    bucket_name: str
    region: str
    profile_name: str = ""
    keys: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_S3_KEYS))

    def key_for(self, data_type: str, market: str, date_str: str) -> str:
        if data_type not in self.keys:
            raise KeyError(f"no S3 key template for data_type {data_type!r}; add it to [s3.keys] in system.toml")
        return self.keys[data_type].format(market=market, date=date_str)


class SchedulerConfig(BaseModel):
    timezone: str


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storage: StorageConfig
    kdb: KdbConfig
    bloomberg: BloombergConfig
    s3: S3Config
    scheduler: SchedulerConfig
    markets: dict[str, MarketConfig]

    def market(self, market: str) -> MarketConfig:
        if market not in self.markets:
            raise KeyError(f"market {market!r} not configured; known markets: {sorted(self.markets)}")
        return self.markets[market]

    def available_data_types(self, market: str) -> list[str]:
        return self.market(market).available_data_types
