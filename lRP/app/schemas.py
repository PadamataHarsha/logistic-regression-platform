import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: list[float] = Field(min_length=1, max_length=20)

    @field_validator("features")
    @classmethod
    def values_must_be_finite(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Feature values must be finite numbers")
        return values