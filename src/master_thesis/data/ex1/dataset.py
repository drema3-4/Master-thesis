import random
import numpy as np
from pathlib import Path

from master_thesis.schemas.ex1.schemas import(
    ObservationsGeneratorParams,
    ObservationsGeneratorOutput,
    DatasetGeneratorParams,
    Trial,
    DatasetItem
)
from master_thesis.data.ex1.laws import h0, h1
from master_thesis.data.ex1.fitting import (
    fit_h0, fit_h1
)
from master_thesis.metrics.evidence import (
    sse,
    bic,
    log_likelihood,
    monte_carlo
)


def gen_observations(
    schema: ObservationsGeneratorParams,
    rng: random.Random
) -> ObservationsGeneratorOutput:
    xs = np.linspace(
        schema.x_min,
        schema.x_max,
        schema.n_observations
    )
    ground_truth = [
        h1(k=schema.k, x=x, alpha=schema.alpha)
        for x in xs
    ]

    observations = []
    for y in ground_truth:
        observations.append(
            y 
            + rng.normalvariate(schema.mu, schema.sigma)
        )

    return ObservationsGeneratorOutput(
        xs=xs,
        ground_truth=ground_truth,
        observations=observations
    )

def trial(
    schema: ObservationsGeneratorParams,
    rng: random.Random
) -> Trial:
    out = gen_observations(schema=schema, rng=rng)
    xs = out.xs
    ground_truth = out.ground_truth
    observations = out.observations

    h0_fit_params = fit_h0(xs, observations)
    h0_predicted = [
        h0(k=h0_fit_params.k, x=x)
        for x in xs
    ]
    sse_h0 = sse(observations, h0_predicted)
    bic_h0 = bic(len(observations), sse_h0, 1-1)

    h1_fit_params = fit_h1(xs, observations)
    h1_predicted = [
        h1(k=h1_fit_params.k, alpha=h1_fit_params.alpha, x=x)
        for x in xs
    ]
    sse_h1 = sse(observations, h1_predicted)
    bic_h1 = bic(len(observations), sse_h1, 2-1)

    delta_bic = bic_h0 - bic_h1

    log_likelihood_h0 = log_likelihood(
        truths=observations,
        predicteds=h0_predicted,
        sigma=schema.sigma
    )
    log_likelihood_h1 = log_likelihood(
        truths=observations,
        predicteds=h1_predicted,
        sigma=schema.sigma
    )
    log_likelihood_ratio = log_likelihood_h1 - log_likelihood_h0

    return Trial(
        xs=xs,
        ground_truth=ground_truth,
        observations=observations,
        h0_fit_params=h0_fit_params,
        h0_predicted=h0_predicted,
        h1_fit_params=h1_fit_params,
        h1_predicted=h1_predicted,
        sse_h0=sse_h0,
        sse_h1=sse_h1,
        bic_h0=bic_h0,
        bic_h1=bic_h1,
        delta_bic=delta_bic,
        log_likelihood_h0=log_likelihood_h0,
        log_likelihood_h1=log_likelihood_h1,
        log_likelihood_ratio=log_likelihood_ratio
    )

def gen_dataset(
    schema: DatasetGeneratorParams
) -> list[DatasetItem]:
    rng = random.Random(schema.seed)
    
    relative_noise_intensity = schema.relative_noise_intensity
    k = schema.k
    relative_unlinear_intensity = schema.relative_unlinear_intensity
    n_observations = schema.n_observations
    mu = schema.mu

    dataset = []
    for X in schema.Xs:
        x_max = X
        x_min = -X
        for s in relative_noise_intensity:
            sigma = s * k * X
            for r in relative_unlinear_intensity:
                alpha = abs((r * k * X) / (X**3))

                delta_bics = []
                for _ in range(1000):
                    trial_ = trial(
                        schema=ObservationsGeneratorParams(
                            n_observations=n_observations,
                            x_max=x_max,
                            x_min=x_min,
                            k=k,
                            alpha=alpha,
                            mu=mu,
                            sigma=sigma
                        ),
                        rng=rng
                    )

                    delta_bics.append(trial_.delta_bic)
                monte_carlo_ = monte_carlo(delta_bics=delta_bics)

                trial_ = trial(
                    schema=ObservationsGeneratorParams(
                        n_observations=n_observations,
                        x_max=x_max,
                        x_min=x_min,
                        k=k,
                        alpha=alpha,
                        mu=mu,
                        sigma=sigma
                    ),
                    rng=rng
                )

                dataset.append(DatasetItem(
                    **trial_.model_dump(),
                    monte_carlo=monte_carlo_,      
                    s=s,
                    r=r,
                    k=k,
                    X=X
                ))

    return dataset

def load_dataset(path: Path) -> list[DatasetItem]:
    dataset = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = DatasetItem.model_validate_json(line)
            dataset.append(item)

    return dataset

def save_dataset(
    datataset: list[DatasetItem],
    path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for item in datataset:
            file.write(item.model_dump_json())
            file.write("\n")