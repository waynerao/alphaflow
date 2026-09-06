from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

AlphaType = Literal["high", "mid", "low", "rtn"]
ALPHA_TYPES: tuple[AlphaType, ...] = ("high", "mid", "low", "rtn")


class RegistryEntry(BaseModel):
    alpha_id: str
    alpha_type: AlphaType
    config_file: str
    active: bool


class RegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alphas: list[RegistryEntry]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "RegistryConfig":
        seen = set()
        duplicates = {e.alpha_id for e in self.alphas if e.alpha_id in seen or seen.add(e.alpha_id)}
        if duplicates:
            raise ValueError(f"duplicate alpha_id(s) in registry: {sorted(duplicates)}")
        return self

    def get_active(self) -> list[RegistryEntry]:
        return [e for e in self.alphas if e.active]

    def get_by_id(self, alpha_id: str) -> RegistryEntry | None:
        return next((e for e in self.alphas if e.alpha_id == alpha_id), None)

    def get_by_type(self, alpha_type: AlphaType) -> list[RegistryEntry]:
        return [e for e in self.alphas if e.alpha_type == alpha_type]
