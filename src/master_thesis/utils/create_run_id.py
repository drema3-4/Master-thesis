from uuid import uuid4


def create_run_id(
    experiment_id: str,
    timestamp: str
) -> str:
    random_suffix = uuid4().hex[:8]

    return f"{experiment_id}_{timestamp}_{random_suffix}"