from pathlib import Path
import argparse

from master_thesis.schemas.ex1.schemas import (
    Ex1Config
)
from master_thesis.utils.load_yaml_config import (
    load_yaml_config
)
from master_thesis.utils.download_model import (
    download_model
)
from master_thesis.pipelines.ex1.make_dataset import (
    make_dataset
)
from master_thesis.pipelines.ex1.experiment import (
    experiment
)


def run_experiment(
    experiment_config_path: str
) -> None:
    experiment_config_path = Path(experiment_config_path)
    
    experiment_config = load_yaml_config(
        path=experiment_config_path,
        schema_type=Ex1Config
    )
    
    if not experiment_config.llm_config.local_dir.exists():
        print("="*40)
        print("You are haven't model weights.")
        print("Donwload weights.")
        download_model(
            experiment_config=experiment_config
        )
        print("Weights downloaded")
        print("Start Docker, start container by compose.yaml and restart script")
        print("="*40)
        return
    
    if not experiment_config.dataset_path.exists():
        make_dataset(
            experiment_config=experiment_config
        )
    
    experiment(
        experiment_config=experiment_config
    )


parser = argparse.ArgumentParser()

parser.add_argument("experiment_config_path")

args = parser.parse_args()

run_experiment(
    experiment_config_path=args.experiment_config_path
)