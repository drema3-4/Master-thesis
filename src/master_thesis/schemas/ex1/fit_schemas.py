from dataclasses import dataclass


@dataclass
class FitH0OutputSchema:
    k: float

@dataclass
class FitH1OutputSchema:
    k: float
    alpha: float