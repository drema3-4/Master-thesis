from openai import OpenAI
from pathlib import Path
import yaml

from master_thesis.schemas.llm_schemas import (
    LLMInitSchema,
    LLMConfigShema
)
from master_thesis.llm.client import LocalLLM
from master_thesis.llm.build_messages import (
    build_messages
)
from master_thesis.schemas.ex1.trial_schema import (
    TrialSchema
)
from master_thesis.data.ex1.load_dataset import (
    load_dataset
)
from master_thesis.llm.build_messages import (
    build_messages
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


llm_init_path = Path(f"{PROJECT_ROOT}/configs/models/llm_init.yaml")
with llm_init_path.open("r", encoding="utf-8") as file:
    llm_init = yaml.safe_load(file)

llm = LocalLLM(**llm_init)


llm_config_path = Path(f"{PROJECT_ROOT}/configs/models/llm_config.yaml")
with llm_config_path.open("r", encoding="utf-8") as file:
    llm_config = yaml.safe_load(file)


dataset_path = Path("data/generated/ex1/dataset.jsonl")
dataset = load_dataset(dataset_path)


for sample in dataset:
    xs = sample.xs
    observations = sample.observations
    H0 = "f(x) = kx"
    H1 = "f(x) = kx + \alpha x^{3}"
    bic_h0 = sample.bic_h0
    sse_h0 = sample.sse_h0
    bic_h1 = sample.bic_h1
    sse_h1 = sample.sse_h1
    delta_bic = bic_h1 - bic_h0

    messages = build_messages(
        TrialSchema(
            xs=xs,
            observations=observations,
            H0=H0,
            H1=H1,
            bic_h0=bic_h0,
            sse_h0=sse_h0,
            bic_h1=bic_h1,
            sse_h1=sse_h1,
            delta_bic=delta_bic
        )
    )

    response = llm.chat(
        messages=messages,
        cfg=llm_config
    )

    print(response["content"])