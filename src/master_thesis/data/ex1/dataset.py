import random
import numpy as np
from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel

from master_thesis.schemas.ex1.schemas import(
    ObservationsGeneratorParams,
    ObservationsGeneratorOutput,
    DatasetCalibrationParams,
    DatasetCalibrationItem,
    ChooseDatasetParamsItem,
    PreItem,
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


SchemaT = TypeVar("SchemaT", bound=BaseModel)


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

def pre_item(
    schema: ObservationsGeneratorParams,
    rng: random.Random
) -> PreItem:
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
    likelihood_ratio = log_likelihood_h1 - log_likelihood_h0

    return PreItem(
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
        likelihood_ratio=likelihood_ratio
    )

def calibration_dataset_parameters_by_monte_carlo(
    schema: DatasetCalibrationParams
) -> list[DatasetCalibrationItem]:
    rng = random.Random(schema.seed)

    n_observations = schema.n_observations
    k = schema.k
    mu = schema.mu
    relative_unlinear_intensity = schema.relative_unlinear_intensity
    relative_noise_intensity = schema.relative_noise_intensity

    calibration_dataset = []
    for X in schema.Xs:
        x_max = X
        x_min = -X
        for s in relative_noise_intensity:
            sigma = s * k * X
            for r in relative_unlinear_intensity:
                alpha = abs((r * k * X) / (X**3))

                delta_bics = []
                for _ in range(1000):
                    pre_item_ = pre_item(
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

                    delta_bics.append(pre_item_.delta_bic)
                monte_carlo_ = monte_carlo(delta_bics=delta_bics)

                calibration_dataset.append(
                    DatasetCalibrationItem(
                        seed=schema.seed,
                        X=X,
                        n_observations=n_observations,
                        k=k,
                        mu=mu,
                        r=r,
                        s=s,
                        monte_carlo=monte_carlo_
                    )
                )

    return calibration_dataset

def choose_dataset_params(
    calibration_dataset_path: Path,
    target_evidence_strength: list[float]
) -> list[ChooseDatasetParamsItem]:
    calibration_dataset = load_dataset(
        path=calibration_dataset_path,
        schema_type=DatasetCalibrationItem
    )

    choose_dataset_params = []
    for level in target_evidence_strength:
        min_diff = abs(level - calibration_dataset[0].monte_carlo)
        index = 0
        
        for i, item in enumerate(calibration_dataset):
            diff = abs(level - item.monte_carlo)

            if diff < min_diff:
                min_diff = diff
                index = i

        item = calibration_dataset[index]

        choose_dataset_params.append(
            ChooseDatasetParamsItem(
                target_evidence_strength=level,
                seed=item.seed,
                X=item.X,
                n_observations=item.n_observations,
                k=item.k,
                mu=item.mu,
                r=item.r,
                s=item.s,
                calibrated_evidence_strength=item.monte_carlo
            )
        )

    return choose_dataset_params

def gen_dataset(
    choose_dataset_params_path: Path
) ->  list[DatasetItem]:
    choose_dataset_params = load_dataset(
        path=choose_dataset_params_path,
        schema_type=ChooseDatasetParamsItem
    )

    dataset = []
    for item in choose_dataset_params:
        x_max = item.X
        x_min = -item.X
        alpha = abs((item.r * item.k * item.X) / (item.X**3))
        sigma = item.s * item.k * item.X

        for _ in range(100):
            rng = random.Random(item.seed)

            pre_item_ = pre_item(
                schema=ObservationsGeneratorParams(
                    n_observations=item.n_observations,
                    x_max=x_max,
                    x_min=x_min,
                    k=item.k,
                    alpha=alpha,
                    mu=item.mu,
                    sigma=sigma
                ),
                rng=rng
            )

            dataset.append(
                DatasetItem(
                    target_evidence_strength=item.target_evidence_strength,
                    **pre_item_.model_dump(),
                    calibrated_evidence_strength=item.calibrated_evidence_strength,      
                    s=item.s,
                    r=item.r,
                    k=item.k,
                    X=item.X
                )
            )

    return dataset

def load_dataset(
    path: Path,
    schema_type: type[SchemaT]
) -> list[DatasetItem] | list[DatasetCalibrationItem] | list[ChooseDatasetParamsItem]:
    dataset = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = schema_type.model_validate_json(line)
            dataset.append(item)

    return dataset

def save_dataset(
    datataset: list[DatasetItem] | list[DatasetCalibrationItem] | list[ChooseDatasetParamsItem],
    path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for item in datataset:
            file.write(item.model_dump_json())
            file.write("\n")