from dataclasses import dataclass


@dataclass
class LLMInitSchema:
    base_url: str
    api_key: str
    model: str

@dataclass
class LLMConfigShema:
    temperature: float
    top_p: float
    top_k: float
    max_tokens: int
    seed: int
    enable_thinking: bool