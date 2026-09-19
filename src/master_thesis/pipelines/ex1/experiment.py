from master_thesis.schemas.ex1.schemas import (
    Ex1Config,
    ExperimentRunItemResult
)
from master_thesis.llm.local_llm import (
    LocalLLMClient
)
from master_thesis.data.ex1.dataset import (
    load_dataset
)
from master_thesis.llm.build_trial import (
    build_trial
)
from master_thesis.llm.schemas import Trial
from master_thesis.utils.parse_hypothesis import (
    parse_hypothesis
)


def experiment(
    experiment_config: Ex1Config
) -> None:
    client = LocalLLMClient(experiment_config.client_config)

    H0 = experiment_config.H0
    H1 = experiment_config.H1

    system_prompt = (
        experiment_config
        .system_prompt_path
        .read_text(encoding="utf-8")
    )
    neutral_prompt = (
        experiment_config
        .neutral_prior_prompt_path
        .read_text(encoding="utf-8")
    )
    wrong_prompt = (
        experiment_config
        .wrong_prior_prompt_path
        .read_text(encoding="utf-8")
    )
    prior_prompts = {
        "neutral": neutral_prompt,
        "wrong": wrong_prompt
    }

    dataset = load_dataset(experiment_config.dataset_path)

    run_id = 1
    experiment_run_result = []
    for type_prior, prior_prompt in prior_prompts.items():
        for item in dataset:
            response = client.chat(
                messages=build_trial(
                    Trial(
                        system_prompt=system_prompt,
                        prior_prompt=prior_prompt,
                        xs=item.xs,
                        observations=item.observations,
                        H0=H0,
                        H1=H1,
                        bic_h0=item.bic_h0,
                        sse_h0=item.sse_h0,
                        bic_h1=item.bic_h1,
                        sse_h1=item.sse_h1,
                        delta_bic=item.bic_h0-item.bic_h1
                    )
                ),
                config=experiment_config.generation_config
            )

            is_right_answer = (
                True 
                if H1 in parse_hypothesis(response["content"])
                else False
            )

            experiment_run_result.append(
                ExperimentRunItemResult(
                    run_id=run_id,
                    H0=H0,
                    H1=H1,
                    right_hypothesis=H1,
                    s=item.s,
                    r=item.r,
                    k=item.k,
                    X=item.X,
                    sse_h0=item.sse_h0,
                    sse_h1=item.sse_h1,
                    bic_h0=item.bic_h0,
                    bic_h1=item.bic_h1,
                    delta_bic=item.bic_h0-item.bic_h1,
                    prior=type_prior,
                    llm_output=response["content"],
                    is_right=is_right_answer
                )
            )

            run_id += 1

    result_path = experiment_config.save_run_experiment_path
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        for item in experiment_run_result:
            file.write(item.model_dump_json())
            file.write("\n")