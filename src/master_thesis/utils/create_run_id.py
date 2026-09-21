from datetime import datetime, timezone
from uuid import uuid4


def create_run_id(experiment_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    random_suffix = uuid4().hex[:8]

    return f"{experiment_id}_{timestamp}_{random_suffix}"