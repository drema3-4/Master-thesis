import yaml
from pathlib import Path

from master_thesis.schemas.ex1.schemas import (
    Ex1Config
)


def make_resolver_config(
    run_dir: Path,
    experiment_config: Ex1Config
) -> None:
    resolver_config = experiment_config.model_dump(mode="json")
    resolver_config["system_prompt"] = (
        experiment_config
        .system_prompt_path
        .read_text(encoding="utf-8")
    )
    prior_prompts = {}
    for type, prior_prompt_path in experiment_config.prior_prompts_paths.items():
        prior_prompts[type] = (
            prior_prompt_path
            .read_text(encoding="utf-8")
        )
    resolver_config["prior_prompts"] = prior_prompts

    path = Path(run_dir / "resolver_config.yaml")

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            resolver_config,
            file,
            allow_unicode=True,
            sort_keys=False
        )