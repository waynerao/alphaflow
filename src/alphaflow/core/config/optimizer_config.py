from pydantic import BaseModel, ConfigDict, Field, model_validator


class MeanVarianceConfig(BaseModel):
    max_single_alpha_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    min_single_alpha_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_aversion: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "MeanVarianceConfig":
        if self.min_single_alpha_weight > self.max_single_alpha_weight:
            raise ValueError(f"min_single_alpha_weight {self.min_single_alpha_weight} exceeds max {self.max_single_alpha_weight}")
        return self

    def feasible_for(self, n_alphas: int) -> bool:
        """sum(w)=1 is only satisfiable if the box constraints can bracket 1.0."""
        return self.min_single_alpha_weight * n_alphas <= 1.0 <= self.max_single_alpha_weight * n_alphas


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean_variance: MeanVarianceConfig
    risk_model: str
