import random
import numpy as np

from master_thesis.schemas.ex1.generator_schemas import (
    GeneratorParamsSchema,
    GeneratorOutputSchema
)
from master_thesis.data.ex1.physical_laws import h1


def gen_observations(
    n_observations: int,
    x_max: float,
    x_min: float,
    k: float,
    alpha: float,
    mu: float,
    sigma: float,
    seed: int
) -> GeneratorOutputSchema:
    n_observations = int(n_observations)
    x_min = float(x_min)
    x_max = float(x_max)
    k = float(k)
    alpha = float(alpha)
    mu = float(mu)
    sigma = float(sigma)
    seed = int(seed)

    random.seed(seed)

    xs = np.linspace(x_min, x_max, n_observations)
    ground_truth = [h1(k=k, x=x, alpha=alpha) for x in xs]

    observations = []
    for y in ground_truth:
        observations.append(y + random.normalvariate(mu, sigma))

    return GeneratorOutputSchema(
        xs,
        ground_truth,
        observations
    )