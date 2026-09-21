from pydantic import BaseModel
from pathlib import Path


class ModelConfig(BaseModel):
    repo_id: str
    revision: str
    local_dir: Path

    served_name: str
    runtime_image: str

class ClientConfig(BaseModel):
    base_url: str
    api_key: str
    model: str

class GenerationConfig(BaseModel):
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    seed: int
    enable_thinking: bool

class Trial(BaseModel):
    system_prompt: str
    prior_prompt: str
    xs: list[float]
    observations: list[float]
    H0: str
    H1: str
    sse_h0: float
    sse_h1: float
    bic_h0: float
    bic_h1: float
    delta_bic: float
    log_likelihood_h0: float
    log_likelihood_h1: float
    log_likelihood_ratio: float
    monte_carlo: float