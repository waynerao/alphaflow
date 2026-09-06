from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alphaflow.core.config.alpha_config.base_config import BaseAlphaConfig
from alphaflow.core.config.system_config import validate_hhmm

PRICE_TYPES = ("open", "close", "vwap")
SESSIONS = ("am", "pm")
RETURN_TYPES = ("raw", "hedged")


class PricePoint(BaseModel):
    """open/close: price_type + optional session (am/pm for dual-session markets).
    vwap: price_type='vwap' plus a vwap_start/vwap_end time range."""

    model_config = ConfigDict(extra="forbid")

    price_type: str
    session: str | None = None
    vwap_start: str | None = None
    vwap_end: str | None = None

    @field_validator("price_type")
    @classmethod
    def validate_price_type(cls, v: str) -> str:
        if v not in PRICE_TYPES:
            raise ValueError(f"price_type must be one of {PRICE_TYPES}, got {v!r}")
        return v

    @field_validator("session")
    @classmethod
    def validate_session(cls, v: str | None) -> str | None:
        if v is not None and v not in SESSIONS:
            raise ValueError(f"session must be one of {SESSIONS} or omitted, got {v!r}")
        return v

    @field_validator("vwap_start", "vwap_end")
    @classmethod
    def validate_vwap_times(cls, v: str | None) -> str | None:
        return None if v is None else validate_hhmm(v)

    @model_validator(mode="after")
    def validate_fields(self) -> "PricePoint":
        if self.price_type == "vwap":
            if self.vwap_start is None or self.vwap_end is None:
                raise ValueError("price_type='vwap' requires both vwap_start and vwap_end")
            if self.vwap_start >= self.vwap_end:
                raise ValueError(f"vwap_start {self.vwap_start} must precede vwap_end {self.vwap_end}")
        elif self.vwap_start is not None or self.vwap_end is not None:
            raise ValueError(f"vwap_start/vwap_end are only valid for price_type='vwap', not {self.price_type!r}")
        return self


class RtnAlphaConfig(BaseAlphaConfig):
    entry: PricePoint
    exit: PricePoint
    horizon: int = Field(description="Days between entry and exit. 0 = intraday.")
    return_type: list[str] = Field(min_length=1)

    @field_validator("return_type")
    @classmethod
    def validate_return_type(cls, v: list[str]) -> list[str]:
        invalid = [t for t in v if t not in RETURN_TYPES]
        if invalid:
            raise ValueError(f"return_type entries must be in {RETURN_TYPES}, got {invalid}")
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate entries in return_type: {v}")
        return v

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"horizon must be >= 0, got {v}")
        return v

    @property
    def is_intraday(self) -> bool:
        return self.horizon == 0
