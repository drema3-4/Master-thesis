from pathlib import Path
from huggingface_hub import snapshot_download


from master_thesis.experiments.ex1.config import (
    resolve_experiment_config
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def download_model(
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
    model_config = resolved_config.model_config_


    local_dir = model_config.local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_config.repo_id,
        revision=model_config.revision,
        local_dir=local_dir,
    )

download_model(name_experiment="ex1_v1.yaml")