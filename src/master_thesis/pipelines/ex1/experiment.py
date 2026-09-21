from pathlib import Path
import json

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
from master_thesis.utils.create_run_id import (
    create_run_id
)


def experiment(
    experiment_config: Ex1Config
) -> None:
    run_id = create_run_id("ex1")

    run_dir = Path(
        experiment_config.save_run_experiment_path
        / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    responses_path = run_dir / "responses.jsonl"
    failures_path = run_dir / "failures.jsonl"
    
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

    trial_index = 1
    with (
        responses_path.open("a", encoding="utf-8") as responses_file,
        failures_path.open("a", encoding="utf-8") as failures_file,
    ):
        for type_prior, prior_prompt in prior_prompts.items():
            for item in dataset:
                try:
                    response = client.chat(
                        messages=build_trial(
                            Trial(
                                system_prompt=system_prompt,
                                prior_prompt=prior_prompt,
                                xs=item.xs,
                                observations=item.observations,
                                H0=H0,
                                H1=H1,
                                sse_h0=item.sse_h0,
                                sse_h1=item.sse_h1,
                                bic_h0=item.bic_h0,
                                bic_h1=item.bic_h1,
                                delta_bic=item.delta_bic,
                                log_likelihood_h0=item.log_likelihood_h0,
                                log_likelihood_h1=item.log_likelihood_h1,
                                log_likelihood_ratio=item.log_likelihood_ratio,
                                monte_carlo=item.monte_carlo
                            )
                        ),
                        config=experiment_config.generation_config
                    )

                    is_right_answer = (
                        True 
                        if "H1" in parse_hypothesis(response["content"])
                        else False
                    )

                    result = ExperimentRunItemResult(
                        run_id=run_id,
                        trial_index=trial_index,
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

                    responses_file.write(result.model_dump_json())
                    responses_file.write("\n")
                    responses_file.flush()

                except Exception as error:
                    failure = {
                        "run_id": run_id,
                        "trial_index": trial_index,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }

                    failures_file.write(
                        json.dumps(failure, ensure_ascii=False)
                    )
                    failures_file.write("\n")
                    failures_file.flush()
            

                trial_index += 1