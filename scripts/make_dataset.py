from pathlib import Path
import yaml

from master_thesis.data.ex1.gen_dataset import (
    gen_dataset
)
from master_thesis.data.ex1.save_dataset import save_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


config_path = Path(f"{PROJECT_ROOT}/configs/ex1/dataset.yaml")
with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

dataset = gen_dataset(**config)

dataset_path = Path(f"{PROJECT_ROOT}/data/generated/ex1/dataset.jsonl")
dataset_path.parent.mkdir(parents=True, exist_ok=True)
save_dataset(dataset, dataset_path)