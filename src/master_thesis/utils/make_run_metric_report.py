from pathlib import Path
import pandas as pd
import json


def make_run_metric_report(
    run_id: str,
    run_dir: Path
) -> None:
    responses = pd.read_json(run_dir / "responses.jsonl", lines=True)
    failures = pd.read_json(run_dir / "failures.jsonl", lines=True)
    outputs = pd.concat([responses, failures], axis=1)

    neutral = outputs[outputs["prior"] == "neutral"]
    wrong = outputs[outputs["prior"] == "wrong"]

    metrics = {
        "run_id": run_id,
        "total_planned": outputs.shape[0],
        "successful": responses.shape[0],
        "failed": failures.shape[0],
        "invalid_responses": (
            outputs[outputs["answer"] == "invalid"]
            .shape[0]
        ),
        "conditions": {
            "neutral": {
                "n": neutral.shape[0],
                "selected_h1": (
                    neutral[neutral["answer"] == "H1"]
                    .shape[0]
                ),
                "p_h1": (
                    neutral[neutral["answer"] == "H1"].shape[0]
                    * 1.0
                    / neutral.shape[0]
                )
            },
            "wrong": {
                "n": wrong.shape[0],
                "selected_h1": (
                    wrong[wrong["answer"] == "H1"]
                    .shape[0]
                ),
                "p_h1": (
                    wrong[wrong["answer"] == "H1"].shape[0]
                    * 1.0
                    / wrong.shape[0]
                )
            }
        }
    }

    path = run_dir / "metrics.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)