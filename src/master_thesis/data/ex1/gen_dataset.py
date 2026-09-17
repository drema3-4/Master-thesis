from master_thesis.data.ex1.gen_observations import (
    gen_observations, 
    GeneratorOutputSchema
)
from master_thesis.data.ex1.fit import fit_h0, fit_h1
from master_thesis.schemas.ex1.dataset_shemas import (
    DatasetItemShema
)
from master_thesis.data.ex1.physical_laws import h0, h1
from master_thesis.metrics.ex1.sse import sse
from master_thesis.metrics.ex1.bic import bic


def gen_dataset(
    seed: int,
    n_observations: int,
    X: float,
    k: float,
    relative_unlinear_intensity: list[float],
    mu: float,
    relative_noise_intensity: list[float]
) -> list[DatasetItemShema]:
    x_max = X
    x_min = -X

    dataset = []
    for s in relative_noise_intensity:
        sigma = s * k * X
        for r in relative_unlinear_intensity:
            alpha = abs((r * k * X) / (X**3))

            out = gen_observations(
                n_observations=n_observations,
                x_max=x_max,
                x_min=x_min,
                k=k,
                alpha=alpha,
                mu=mu,
                sigma=sigma,
                seed=seed
            )
            xs = out.xs
            ground_truth = out.ground_truth
            observations = out.observations

            h0_fit_params = fit_h0(xs, observations)
            h0_predicted = [h0(**h0_fit_params.__dict__, x=x) for x in xs]
            sse_h0 = sse(observations, h0_predicted)
            bic_h0 = bic(len(observations), sse_h0, len(h0_fit_params.__dict__))

            h1_fit_params = fit_h1(xs, observations)
            h1_predicted = [h1(**h1_fit_params.__dict__, x=x) for x in xs]
            sse_h1 = sse(observations, h1_predicted)
            bic_h1 = bic(len(observations), sse_h1, len(h1_fit_params.__dict__))

            dataset.append(DatasetItemShema(
                xs=xs,
                ground_truth=ground_truth,
                observations=observations,
                h0_fit_params=h0_fit_params,
                h0_predicted=h0_predicted,
                sse_h0=sse_h0,
                bic_h0=bic_h0,
                h1_fit_params=h1_fit_params,
                h1_predicted=h1_predicted,
                sse_h1=sse_h1,
                bic_h1=bic_h1,                
                s=s,
                r=r,
                k=k,
                X=X
            ))

    return dataset