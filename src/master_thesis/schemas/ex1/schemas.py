from pydantic import BaseModel
from pathlib import Path

from master_thesis.llm.schemas import (
    ModelConfig,
    ClientConfig,
    GenerationConfig
)


class ObservationsGeneratorParams(BaseModel):
    n_observations: int
    x_max: float
    x_min: float
    k: float
    alpha: float
    mu: float
    sigma: float
    seed: int

class ObservationsGeneratorOutput(BaseModel):
    xs: list[float]
    ground_truth: list[float]
    observations: list[float]

class FitH0Output(BaseModel):
    k: float

class FitH1Output(BaseModel):
    k: float
    alpha: float

class DatasetGeneratorParams(BaseModel):
    seed: int
    n_observations: int
    X: float
    k: float
    relative_unlinear_intensity: list[float]
    mu: float
    relative_noise_intensity: list[float]

class DatasetItem(BaseModel):
    xs: list[float]
    ground_truth: list[float]
    observations: list[float]
    h0_fit_params: FitH0Output
    h0_predicted: list[float]
    sse_h0: float
    bic_h0: float
    h1_fit_params: FitH1Output
    h1_predicted: list[float]
    sse_h1: float
    bic_h1: float
    s: float
    r: float
    k: float
    X: float

class ExperimentRunItemResult(BaseModel):
    run_id: int
    H0: str
    H1: str
    right_hypothesis: str
    s: float
    r: float
    k: float
    X: float
    sse_h0: float
    sse_h1: float
    bic_h0: float
    bic_h1: float
    delta_bic: float
    prior: str
    llm_output: str
    is_right: bool

class Ex1Config(BaseModel):
    H0: str
    H1: str
    dataset_generation: DatasetGeneratorParams
    dataset_path: Path
    llm_config: ModelConfig
    client_config: ClientConfig
    generation_config: GenerationConfig
    system_prompt_path: Path
    neutral_prior_prompt_path: Path
    wrong_prior_prompt_path: Path
    save_run_experiment_path: Path