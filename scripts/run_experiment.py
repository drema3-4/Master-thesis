from pathlib import Path
import argparse
from datetime import datetime, timezone

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
from master_thesis.utils.create_run_id import (
    create_run_id
)
from master_thesis.utils.make_resolver_config import (
    make_resolver_config
)
from master_thesis.utils.make_manifest import (
    make_manifest
)
from master_thesis.utils.make_run_metric_report import (
    make_run_metric_report
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

    started_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    run_id = create_run_id(
        experiment_id="ex1",
        timestamp=started_time
    )
    run_dir = Path(
        experiment_config.save_run_experiment_path
        / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    make_resolver_config(
        run_dir=run_dir,
        experiment_config=experiment_config
    )
        
    successful_trials, failed_trials = experiment(
        experiment_config=experiment_config,
        run_id=run_id,
        run_dir=run_dir
    )

    finished_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    make_manifest(
        run_id=run_id,
        run_dir=run_dir,
        experiment_config=experiment_config,
        started_at=started_time,
        finished_at=finished_time,
        successful_trials=successful_trials,
        failed_trials=failed_trials
    )

    make_run_metric_report(
        run_id=run_id,
        run_dir=run_dir
    )


parser = argparse.ArgumentParser()

parser.add_argument("experiment_config_path")

args = parser.parse_args()

run_experiment(
    experiment_config_path=args.experiment_config_path
)