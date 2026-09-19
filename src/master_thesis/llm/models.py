from pydantic import BaseModel
from pathlib import Path


class ModelConfigSchema(BaseModel):
    repo_id: str
    revision: str
    local_dir: Path

    served_name: str
    runtime_image: str

class LLMClientConfigSchema(BaseModel):
    base_url: str
    api_key: str
    model: str

class GenerationConfigSchema(BaseModel):
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    seed: int
    enable_thinking: bool