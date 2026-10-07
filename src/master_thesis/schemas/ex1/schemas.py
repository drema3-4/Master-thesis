from pydantic import BaseModel, Field
from pathlib import Path
from typing import Literal

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
    true_hypothesis: Literal["H0", "H1"] = "H1"

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
    monte_carlo_h0: float | None = None
    monte_carlo_h1: float | None = None

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
    calibrated_evidence_strength_h0: float | None = None
    calibrated_evidence_strength_h1: float | None = None

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
    log_likelihood_ratio: float

class DatasetItem(BaseModel):
    dataset_id: str
    regime_id: str
    replicate_index: int
    generation_seed: int
    right_hypothesis: Literal["H0", "H1"] = "H1"
    evidence_calibrated_for: Literal["H1"] = "H1"
    target_evidence_strength: float | None
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
    log_likelihood_ratio: float
    calibrated_evidence_strength: float | None
    s: float
    r: float
    k: float
    X: float

class ExperimentRunItemResult(BaseModel):
    run_id: str
    trial_index: int
    dataset_id: str
    regime_id: str
    replicate_index: int
    generation_seed: int
    H0: str
    H1: str
    right_hypothesis: Literal["H0", "H1"]
    hypothesis_order: Literal["H0_H1", "H1_H0"]
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
    log_likelihood_ratio: float
    target_evidence_strength: float | None
    calibrated_evidence_strength: float | None
    prior: str
    prior_h0: float
    prior_h1: float
    expected_p_h1: float
    expected_choice: str
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
    dataset_seed: int
    n_datasets_per_level: int
    true_hypotheses: list[Literal["H0", "H1"]] = Field(
        default_factory=lambda: ["H1"],
        min_length=1
    )
    hypothesis_orders: list[Literal["H0_H1", "H1_H0"]] = Field(
        default_factory=lambda: ["H0_H1"],
        min_length=1
    )
    llm_config: ModelConfig
    client_config: ClientConfig
    generation_config: GenerationConfig
    system_prompt_path: Path
    prior_prompts_paths: dict[str, Path]
    prior_probabilities: dict[
        str,
        dict[str, float] | dict[str, dict[str, float]]
    ]
    save_run_experiment_path: Path
