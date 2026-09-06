from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alphaflow.core.alpha_base.market import Market
from alphaflow.core.utils.dates import to_date


class ModelConfig(BaseModel):
    """A model = market + a set of alphas + a shared rtn_alpha_id (which alone carries
    entry/exit timing) + the weights OptimizerRunner writes back."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    market: Market
    alpha_ids: list[str] = Field(min_length=1)
    rtn_alpha_id: str
    start_date: str
    end_date: str
    save_to_parquet: bool = False
    alpha_weights: dict[str, float] = Field(default_factory=dict)

    # Set by the loader so validate_model_id_matches_filename() has something to compare against
    source_path: str | None = Field(default=None, exclude=True)

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        to_date(v)  # raises ValueError on anything that is not YYYYMMDD
        return v

    @model_validator(mode="after")
    def validate_date_order(self) -> "ModelConfig":
        if to_date(self.end_date) < to_date(self.start_date):
            raise ValueError(f"end_date {self.end_date} precedes start_date {self.start_date}")
        return self

    @model_validator(mode="after")
    def validate_unique_alpha_ids(self) -> "ModelConfig":
        if len(set(self.alpha_ids)) != len(self.alpha_ids):
            raise ValueError(f"duplicate entries in alpha_ids: {self.alpha_ids}")
        return self

    @model_validator(mode="after")
    def validate_model_id_matches_filename(self) -> "ModelConfig":
        if self.source_path is not None and Path(self.source_path).stem != self.model_id:
            raise ValueError(f"model_id {self.model_id!r} does not match filename stem {Path(self.source_path).stem!r}")
        return self

    @model_validator(mode="after")
    def validate_weights_keys(self) -> "ModelConfig":
        """Empty means 'not yet optimized'. Non-empty must cover exactly alpha_ids -
        a stale weight for a removed alpha would silently skew the composite."""
        if self.alpha_weights and set(self.alpha_weights) != set(self.alpha_ids):
            missing = sorted(set(self.alpha_ids) - set(self.alpha_weights))
            extra = sorted(set(self.alpha_weights) - set(self.alpha_ids))
            raise ValueError(f"alpha_weights keys must match alpha_ids exactly (missing={missing}, unexpected={extra})")
        return self

    @property
    def is_optimized(self) -> bool:
        return bool(self.alpha_weights)

    def weight(self, alpha_id: str) -> float:
        return self.alpha_weights.get(alpha_id, 0.0)
