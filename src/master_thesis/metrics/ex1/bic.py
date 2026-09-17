import numpy as np


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