from dataclasses import dataclass


@dataclass
class GeneratorParamsSchema:
    n_observations: int
    x_max: float
    x_min: float
    k: float
    alpha: float
    mu: float
    sigma: float
    seed: int

@dataclass
class GeneratorOutputSchema:
    xs: list[float]
    ground_truth: list[float]
    observations: list[float]