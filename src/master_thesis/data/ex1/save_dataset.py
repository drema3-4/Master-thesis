from pathlib import Path

from master_thesis.schemas.ex1.dataset_shemas import (
    DatasetItemShema
)


def save_dataset(
    datataset: list[DatasetItemShema],
    path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for item in datataset:
            file.write(item.model_dump_json())
            file.write("\n")