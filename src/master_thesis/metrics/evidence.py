import numpy as np


def sse(
    truths: list[float],
    predicteds: list[float]
) -> float:
    sse = 0

    for truth, predicted in zip(truths, predicteds):
        diff = truth - predicted
        sse += diff * diff

    return float(sse)

def bic(
    n_observations: int,
    sse: float,
    n_params: int
) -> float:
    n = float(n_observations)
    sse = float(sse)
    p = float(n_params)

    bic = n * np.log(sse / n) + p * np.log(n)

    return bic

def log_likelihood(
    truths: list[float],
    predicteds: list[float],
    sigma: float
) -> float:
    n = len(truths)

    sse_ = sse(truths=truths, predicteds=predicteds)

    return (
        (-n / 2.0)
        * np.log(2.0 * np.pi * sigma**2)
        - (sse_ / (2.0 * sigma**2))
    )

def monte_carlo(
    delta_bics: list[float]
) -> float:
    n = len(delta_bics)

    metric = 0.0
    for delta_bic in delta_bics:
        if delta_bic > 0:
            metric += 1.0

    return metric / n