from pydantic import BaseModel


class ObservationsGeneratorParamsSchema(BaseModel):
    n_observations: int
    x_max: float
    x_min: float
    k: float
    alpha: float
    mu: float
    sigma: float
    seed: int

class ObservationsGeneratorOutputSchema(BaseModel):
    xs: list[float]
    ground_truth: list[float]
    observations: list[float]

class FitH0OutputSchema(BaseModel):
    k: float

class FitH1OutputSchema(BaseModel):
    k: float
    alpha: float

class DatasetGeneratorParamsSchema(BaseModel):
    seed: int
    n_observations: int
    X: float
    k: float
    relative_unlinear_intensity: list[float]
    mu: float
    relative_noise_intensity: list[float]

class DatasetItemShema(BaseModel):
    xs: list[float]
    ground_truth: list[float]
    observations: list[float]
    h0_fit_params: FitH0OutputSchema
    h0_predicted: list[float]
    sse_h0: float
    bic_h0: float
    h1_fit_params: FitH1OutputSchema
    h1_predicted: list[float]
    sse_h1: float
    bic_h1: float
    s: float
    r: float
    k: float
    X: float

class TrialSchema(BaseModel):
    system_prompt: str
    prior_prompt: str
    xs: list[float]
    observations: list[float]
    H0: str
    H1: str
    bic_h0: float
    sse_h0: float
    bic_h1: float
    sse_h1: float
    delta_bic: float