from dataclasses import dataclass


@dataclass
class TrialSchema:
    xs: list[float]
    observations: list[float]
    H0: str
    H1: str
    bic_h0: float
    sse_h0: float
    bic_h1: float
    sse_h1: float
    delta_bic: float