from pathlib import Path
from pydantic import BaseModel, Field

from master_thesis.utils.load_yaml_config import (
    load_yaml_config
)
from master_thesis.experiments.ex1.models import (
    DatasetGeneratorParamsSchema,
)
from master_thesis.llm.models import (
    ModelConfigSchema,
    GenerationConfigSchema,
    LLMClientConfigSchema,
)


class Ex1ExperimentConfigSchema(BaseModel):
    experiment_id: str

    dataset_config_path: Path
    dataset_path: Path

    model_config_path: Path
    client_config_path: Path
    generation_config_path: Path

    prompt_version: str

    llm_trials_per_condition: int = Field(ge=1)

class ResolvedEx1ConfigSchema(BaseModel):
    experiment: Ex1ExperimentConfigSchema

    experiment_id: str

    dataset_config: DatasetGeneratorParamsSchema
    dataset_path: Path

    model_config_: ModelConfigSchema
    client_config: LLMClientConfigSchema
    generation_config: GenerationConfigSchema

    prompt_version: str
    system_prompt: str
    prior_prompts: dict[str, str]

    llm_trials_per_condition: int


def resolve_experiment_config(
    project_root: Path,
    experiment_config_path: Path,
) -> ResolvedEx1ConfigSchema:
    experiment = load_yaml_config(
        experiment_config_path,
        Ex1ExperimentConfigSchema,
    )

    experiment_id = experiment.experiment_id

    dataset_config = load_yaml_config(
        project_root / experiment.dataset_config_path,
        DatasetGeneratorParamsSchema,
    )
    dataset_path = project_root / experiment.dataset_path

    model_config_ = load_yaml_config(
        project_root / experiment.model_config_path,
        ModelConfigSchema
    )
    model_config_ = model_config_.model_copy(
        update={
            "local_dir": (
                project_root / model_config_.local_dir
            ).resolve()
        }
    )
    client_config = load_yaml_config(
        project_root / experiment.client_config_path,
        LLMClientConfigSchema,
    )
    generation_config = load_yaml_config(
        project_root / experiment.generation_config_path,
        GenerationConfigSchema,
    )

    prompt_version = experiment.prompt_version
    system_prompt_path = Path(
        project_root
        / "prompts"
        / "ex1"
        / prompt_version
        / "system.txt"
    )
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    neutral_prior_prompt_path = Path(
        project_root
        / "prompts"
        / "ex1"
        / prompt_version
        / "neutral.txt"
    )
    neutral_prior_prompt = neutral_prior_prompt_path.read_text(encoding="utf-8")
    wrong_prior_prompt_path = Path(
        project_root
        / "prompts"
        / "ex1"
        / prompt_version
        / "wrong.txt"
    )
    wrong_prior_prompt = wrong_prior_prompt_path.read_text(encoding="utf-8")
    prior_prompts = {
        "neutral": neutral_prior_prompt,
        "wrong": wrong_prior_prompt
    }

    llm_trials_per_condition = experiment.llm_trials_per_condition
    
    return ResolvedEx1ConfigSchema(
        experiment=experiment,
        experiment_id=experiment_id,
        dataset_config=dataset_config,
        dataset_path=dataset_path,
        model_config_=model_config_,
        client_config=client_config,
        generation_config=generation_config,
        prompt_version=prompt_version,
        system_prompt=system_prompt,
        prior_prompts=prior_prompts,
        llm_trials_per_condition=llm_trials_per_condition
    )