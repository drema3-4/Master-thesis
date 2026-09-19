from huggingface_hub import snapshot_download

from master_thesis.schemas.ex1.schemas import (
    Ex1Config
)


def download_model(
    experiment_config: Ex1Config
) -> None:
    model_config = experiment_config.llm_config

    local_dir = model_config.local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_config.repo_id,
        revision=model_config.revision,
        local_dir=local_dir,
    )