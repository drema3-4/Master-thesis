from pathlib import Path

from master_thesis.schemas.ex1.dataset_shemas import (
    DatasetItemShema
)


def load_dataset(path: Path) -> list[DatasetItemShema]:
    dataset = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = DatasetItemShema.model_validate_json(line)
            dataset.append(item)

    return dataset