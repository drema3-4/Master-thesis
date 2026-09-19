from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def load_yaml_config(
    path: Path,
    schema_type: type[SchemaT],
) -> SchemaT:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return schema_type.model_validate(data)