import numpy as np


def sse(truths: list[float], predicteds: list[float]) -> float:
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