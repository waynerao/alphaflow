from pydantic import BaseModel, ConfigDict, Field

from alphaflow.core.alpha_base.market import Market
from alphaflow.core.config.registry_config import AlphaType


class BaseAlphaConfig(BaseModel):
    """Shared fields for all alpha types.

    required_inputs are data_type names; they map to raw_data/{market}/{data_type}/ on the
    shared drive and to PITDataManager getters in live mode. The columns consumed inside
    compute() are the Python subclass's business, not this config's.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    category: str
    tags: list[str] = Field(default_factory=list)
    formula: str
    handler_class: str
    alpha_type: AlphaType
    market_scope: list[Market] = Field(min_length=1)
    required_inputs: list[str] = Field(min_length=1)
    custom_exclusions: list[str] = Field(default_factory=list)
