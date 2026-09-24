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

class ObservationsGeneratorOutput(BaseModel):
    xs: list[float]
    ground_truth: list[float]
    observations: list[float]

class FitH0Output(BaseModel):
    k: float

class FitH1Output(BaseModel):
    k: float
    alpha: float

class DatasetCalibrationParams(BaseModel):
    seed: int
    Xs: list[float]
    n_observations: int
    k: float
    mu: float
    relative_unlinear_intensity: list[float]
    relative_noise_intensity: list[float]
    
class DatasetCalibrationItem(BaseModel):
    seed: int
    X: float
    n_observations: int
    k: float
    mu: float
    r: float
    s: float
    monte_carlo: float

class ChooseDatasetParamsItem(BaseModel):
    target_evidence_strength: float
    seed: int
    X: float
    n_observations: int
    k: float
    mu: float
    r: float
    s: float
    calibrated_evidence_strength: float

class PreItem(BaseModel):
    xs: list[float]
    ground_truth: list[float]
    observations: list[float]
    h0_fit_params: FitH0Output
    h0_predicted: list[float]
    h1_fit_params: FitH1Output
    h1_predicted: list[float]
    sse_h0: float
    sse_h1: float
    bic_h0: float
    bic_h1: float
    delta_bic: float
    log_likelihood_h0: float
    log_likelihood_h1: float
    likelihood_ratio: float

class DatasetItem(BaseModel):
    target_evidence_strength: float
    xs: list[float]
    ground_truth: list[float]
    observations: list[float]
    h0_fit_params: FitH0Output
    h0_predicted: list[float]
    h1_fit_params: FitH1Output
    h1_predicted: list[float]
    sse_h0: float
    sse_h1: float
    bic_h0: float
    bic_h1: float
    delta_bic: float
    log_likelihood_h0: float
    log_likelihood_h1: float
    likelihood_ratio: float
    calibrated_evidence_strength: float
    s: float
    r: float
    k: float
    X: float

class ExperimentRunItemResult(BaseModel):
    run_id: str
    trial_index: int
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
    log_likelihood_h0: float
    log_likelihood_h1: float
    likelihood_ratio: float
    target_evidence_strength: float
    calibrated_evidence_strength: float
    prior: str
    llm_output: str
    answer: str
    is_right: bool

class Ex1Config(BaseModel):
    experiment_id: str
    schema_version: str
    H0: str
    H1: str
    dataset_calibration_params: DatasetCalibrationParams
    calibration_dataset_path: Path
    target_evidence_strength: list[float]
    choose_dataset_params_path: Path
    dataset_path: Path
    llm_config: ModelConfig
    client_config: ClientConfig
    generation_config: GenerationConfig
    system_prompt_path: Path
    prior_prompts_paths: dict[str, Path]
    save_run_experiment_path: Path