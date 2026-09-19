from pathlib import Path

from master_thesis.experiments.ex1.config import (
    resolve_experiment_config
)
from master_thesis.llm.client import LocalLLM
from master_thesis.experiments.ex1.dataset import (
    load_dataset
)
from master_thesis.experiments.ex1.prompting import (
    build_messages
)
from master_thesis.experiments.ex1.models import (
    TrialSchema
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_experiment(
    name_experiment: str
) -> None:
    experiment_config_path=(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / {name_experiment}
    )
    resolved_config = resolve_experiment_config(
        project_root=PROJECT_ROOT,
        experiment_config_path=experiment_config_path
    )


    llm = LocalLLM(
        resolved_config.client_config
    )

    dataset = load_dataset(resolved_config.dataset_path)

    system_prompt = resolved_config.system_prompt
    prior_prompts = resolved_config.prior_prompts

    for type_prompt, prior_prompt in prior_prompts.items():
        for sample in dataset:
            xs = sample.xs
            observations = sample.observations
            H0 = "f(x) = kx"
            H1 = r"f(x) = kx + \alpha x^{3}"
            bic_h0 = sample.bic_h0
            sse_h0 = sample.sse_h0
            bic_h1 = sample.bic_h1
            sse_h1 = sample.sse_h1
            delta_bic = bic_h0 - bic_h1

            messages = build_messages(
                TrialSchema(
                    system_prompt=system_prompt,
                    prior_prompt=prior_prompt,
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
                config=resolved_config.generation_config
            )

            print(type_prompt, response["content"])