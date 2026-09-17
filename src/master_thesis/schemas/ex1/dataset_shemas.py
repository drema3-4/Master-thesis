from dataclasses import dataclass
from pydantic import BaseModel

from master_thesis.schemas.ex1.fit_schemas import (
    FitH0OutputSchema,
    FitH1OutputSchema
)


class DatasetItemShema(BaseModel):
    xs: list[float]
    ground_truth: list[float]
    observations: list[float]
    h0_fit_params: FitH0OutputSchema
    h0_predicted: list[float]
    sse_h0: float
    bic_h0: float
    h1_fit_params: FitH1OutputSchema
    h1_predicted: list[float]
    sse_h1: float
    bic_h1: float
    s: float
    r: float
    k: float
    X: float