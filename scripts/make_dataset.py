from pathlib import Path

from master_thesis.experiments.ex1.config import (
    resolve_experiment_config
)
from master_thesis.experiments.ex1.dataset import (
    gen_dataset,
    save_dataset
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_dataset(
    name_experiment: str
) -> None:
    experiment_config_path=(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / name_experiment
    )
    resolved_config = resolve_experiment_config(
        project_root=PROJECT_ROOT,
        experiment_config_path=experiment_config_path
    )


    dataset = gen_dataset(resolved_config.dataset_config)

    dataset_path = resolved_config.dataset_path
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(dataset, dataset_path)

make_dataset(name_experiment="ex1_v1.yaml")