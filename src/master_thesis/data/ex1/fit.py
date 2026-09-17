import numpy as np

from master_thesis.schemas.ex1.fit_schemas import (
    FitH0OutputSchema,
    FitH1OutputSchema
)


def fit_h0(
    xs: list[float],
    observations: list[float]
) -> FitH0OutputSchema:
    return FitH0OutputSchema(k=1.0)


def fit_h1(
    xs: list[float],
    observations: list[float]
) -> FitH1OutputSchema:
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

    return FitH1OutputSchema(k=1.0, alpha=float(alpha))