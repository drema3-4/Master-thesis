import numpy as np

from master_thesis.schemas.ex1.schemas import (
    FitH0Output,
    FitH1Output
)


def fit_h0(
    xs: list[float],
    observations: list[float]
) -> FitH0Output:
    return FitH0Output(k=1.0)

def fit_h1(
    xs: list[float],
    observations: list[float]
) -> FitH1Output:
    xs = np.array(xs)
    observations = np.array(observations)

    residuals = observations - xs
    X = (xs ** 3).reshape(-1, 1)

    params, *_ = np.linalg.lstsq(
        X,
        residuals,
        rcond=None
    )

    alpha = params[0]

    return FitH1Output(k=1.0, alpha=float(alpha))