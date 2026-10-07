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
    if schema.true_hypothesis == "H0":
        ground_truth = [
            h0(k=schema.k, x=x)
            for x in xs
        ]
    else:
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
    log_likelihood_ratio = log_likelihood_h1 - log_likelihood_h0

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
        log_likelihood_ratio=log_likelihood_ratio
    )

def calibration_dataset_parameters_by_monte_carlo(
    schema: DatasetCalibrationParams
) -> list[DatasetCalibrationItem]:
    rng_h1 = random.Random(schema.seed)
    rng_h0 = random.Random(schema.seed + 1000000000)

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

            delta_bics_h0 = []
            for _ in range(1000):
                pre_item_h0 = pre_item(
                    schema=ObservationsGeneratorParams(
                        n_observations=n_observations,
                        x_max=x_max,
                        x_min=x_min,
                        k=k,
                        alpha=0.0,
                        mu=mu,
                        sigma=sigma,
                        true_hypothesis="H0"
                    ),
                    rng=rng_h0
                )
                delta_bics_h0.append(pre_item_h0.delta_bic)
            monte_carlo_h0 = monte_carlo(
                delta_bics=delta_bics_h0,
                right_hypothesis="H0"
            )

            for r in relative_unlinear_intensity:
                alpha = abs((r * k * X) / (X**3))

                delta_bics_h1 = []
                for _ in range(1000):
                    pre_item_ = pre_item(
                        schema=ObservationsGeneratorParams(
                            n_observations=n_observations,
                            x_max=x_max,
                            x_min=x_min,
                            k=k,
                            alpha=alpha,
                            mu=mu,
                            sigma=sigma,
                            true_hypothesis="H1"
                        ),
                        rng=rng_h1
                    )

                    delta_bics_h1.append(pre_item_.delta_bic)
                monte_carlo_h1 = monte_carlo(
                    delta_bics=delta_bics_h1,
                    right_hypothesis="H1"
                )

                calibration_dataset.append(
                    DatasetCalibrationItem(
                        seed=schema.seed,
                        X=X,
                        n_observations=n_observations,
                        k=k,
                        mu=mu,
                        r=r,
                        s=s,
                        monte_carlo=monte_carlo_h1,
                        monte_carlo_h0=monte_carlo_h0,
                        monte_carlo_h1=monte_carlo_h1
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
        first_h1_strength = (
            calibration_dataset[0].monte_carlo_h1
            if calibration_dataset[0].monte_carlo_h1 is not None
            else calibration_dataset[0].monte_carlo
        )
        min_diff = abs(level - first_h1_strength)
        index = 0
        
        for i, item in enumerate(calibration_dataset):
            h1_strength = (
                item.monte_carlo_h1
                if item.monte_carlo_h1 is not None
                else item.monte_carlo
            )
            diff = abs(level - h1_strength)

            if diff < min_diff:
                min_diff = diff
                index = i

        item = calibration_dataset[index]
        h1_strength = (
            item.monte_carlo_h1
            if item.monte_carlo_h1 is not None
            else item.monte_carlo
        )

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
                calibrated_evidence_strength=h1_strength,
                calibrated_evidence_strength_h0=item.monte_carlo_h0,
                calibrated_evidence_strength_h1=h1_strength
            )
        )

    return choose_dataset_params

def gen_dataset(
    choose_dataset_params_path: Path,
    dataset_seed: int,
    n_datasets_per_level: int,
    true_hypotheses: list[str] | None = None
) ->  list[DatasetItem]:
    choose_dataset_params = load_dataset(
        path=choose_dataset_params_path,
        schema_type=ChooseDatasetParamsItem
    )

    if true_hypotheses is None:
        true_hypotheses = ["H1"]

    dataset = []
    for level_index, item in enumerate(choose_dataset_params):
        x_max = item.X
        x_min = -item.X
        alpha = abs((item.r * item.k * item.X) / (item.X**3))
        sigma = item.s * item.k * item.X

        for hypothesis_index, true_hypothesis in enumerate(true_hypotheses):
            regime_id = (
                f"regime_{true_hypothesis.lower()}_"
                f"{level_index + 1:02d}"
            )
            for replicate_index in range(n_datasets_per_level):
                generation_seed = (
                    dataset_seed
                    + hypothesis_index * 10000000
                    + level_index * 100000
                    + replicate_index
                )
                rng = random.Random(generation_seed)

                pre_item_ = pre_item(
                    schema=ObservationsGeneratorParams(
                        n_observations=item.n_observations,
                        x_max=x_max,
                        x_min=x_min,
                        k=item.k,
                        alpha=(
                            alpha
                            if true_hypothesis == "H1"
                            else 0.0
                        ),
                        mu=item.mu,
                        sigma=sigma,
                        true_hypothesis=true_hypothesis
                    ),
                    rng=rng
                )

                dataset.append(
                    DatasetItem(
                        dataset_id=(
                            f"dataset_{true_hypothesis.lower()}_"
                            f"{level_index + 1:02d}_"
                            f"{replicate_index + 1:03d}"
                        ),
                        regime_id=regime_id,
                        replicate_index=replicate_index + 1,
                        generation_seed=generation_seed,
                        right_hypothesis=true_hypothesis,
                        evidence_calibrated_for="H1",
                        target_evidence_strength=(
                            item.target_evidence_strength
                            if true_hypothesis == "H1"
                            else None
                        ),
                        **pre_item_.model_dump(),
                        calibrated_evidence_strength=(
                            (
                                item.calibrated_evidence_strength_h1
                                if item.calibrated_evidence_strength_h1
                                is not None
                                else item.calibrated_evidence_strength
                            )
                            if true_hypothesis == "H1"
                            else item.calibrated_evidence_strength_h0
                        ),
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
