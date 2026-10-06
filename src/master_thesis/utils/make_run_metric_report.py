from pathlib import Path
import pandas as pd
import json

from master_thesis.schemas.ex1.schemas import Ex1Config


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    return pd.read_json(path, lines=True)


def confidence_interval_95(
    selected: int,
    total: int
) -> tuple[float | None, float | None]:
    if total == 0:
        return (None, None)

    z = 1.96
    probability = selected / total
    denominator = 1.0 + z**2 / total
    center = (
        probability + z**2 / (2.0 * total)
    ) / denominator
    margin = (
        z
        * (
            probability * (1.0 - probability) / total
            + z**2 / (4.0 * total**2)
        )**0.5
        / denominator
    )

    return (center - margin, center + margin)


def condition_metrics(condition: pd.DataFrame) -> dict:
    valid = condition[condition["answer"].isin(["H0", "H1"])]
    selected_h1 = valid[valid["answer"] == "H1"].shape[0]
    valid_n = valid.shape[0]
    ci_low, ci_high = confidence_interval_95(selected_h1, valid_n)

    expected_matches = (
        valid[valid["answer"] == valid["expected_choice"]]
        .shape[0]
    )

    return {
        "n": int(condition.shape[0]),
        "valid_n": int(valid_n),
        "invalid": int(condition.shape[0] - valid_n),
        "selected_h1": int(selected_h1),
        "p_h1": (
            selected_h1 / valid_n
            if valid_n > 0
            else None
        ),
        "p_h1_ci95": [ci_low, ci_high],
        "expected_choice_matches": int(expected_matches),
        "p_expected_choice_matches": (
            expected_matches / valid_n
            if valid_n > 0
            else None
        ),
        "mean_expected_p_h1": (
            float(valid["expected_p_h1"].mean())
            if valid_n > 0
            else None
        )
    }


def make_run_metric_report(
    run_id: str,
    run_dir: Path,
    experiment_config: Ex1Config
) -> None:
    responses = read_jsonl(run_dir / "responses.jsonl")
    failures = read_jsonl(run_dir / "failures.jsonl")

    with experiment_config.dataset_path.open("r", encoding="utf-8") as file:
        dataset_size = sum(1 for line in file if line.strip())

    total_planned = (
        dataset_size
        * len(experiment_config.prior_prompts_paths)
    )

    conditions = {}
    evidence_strength = []
    if not responses.empty:
        for prior in experiment_config.prior_prompts_paths:
            condition = responses[responses["prior"] == prior]
            conditions[prior] = condition_metrics(condition)

        group_columns = [
            "prior",
            "target_evidence_strength",
            "calibrated_evidence_strength"
        ]
        for values, group in responses.groupby(
            group_columns,
            sort=True
        ):
            prior, target, calibrated = values
            evidence_strength.append({
                "prior": str(prior),
                "target_evidence_strength": float(target),
                "calibrated_evidence_strength": float(calibrated),
                **condition_metrics(group)
            })

    metrics = {
        "run_id": run_id,
        "model": experiment_config.client_config.model,
        "total_planned": int(total_planned),
        "completed": int(responses.shape[0] + failures.shape[0]),
        "successful": int(responses.shape[0]),
        "failed": int(failures.shape[0]),
        "invalid_responses": (
            int(responses[responses["answer"] == "invalid"].shape[0])
            if not responses.empty
            else 0
        ),
        "conditions": conditions,
        "evidence_strength": evidence_strength
    }

    path = run_dir / "metrics.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
