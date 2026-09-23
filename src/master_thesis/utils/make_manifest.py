from pathlib import Path
import subprocess
import json

from master_thesis.schemas.ex1.schemas import (
    Ex1Config
)


def make_manifest(
    run_id: str,
    run_dir: Path,
    experiment_config: Ex1Config,
    started_at: str,
    finished_at: str,
    successful_trials: int,
    failed_trials: int
) -> None:
    manifest = {
        "experiment_id": experiment_config.experiment_id,
        "schema_version": experiment_config.schema_version,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "git_commit": (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
            ).strip()
        ),
        "model": experiment_config.llm_config.model_dump(mode="json"),
        "dataset_path": str(experiment_config.dataset_path),
        "planned_trials": successful_trials + failed_trials,
        "successful_trials": successful_trials,
        "failed_trials": failed_trials
    }
    
    path = Path(run_dir / "manifest.json")

    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)