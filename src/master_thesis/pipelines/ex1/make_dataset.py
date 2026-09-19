from master_thesis.schemas.ex1.schemas import (
    Ex1Config
)
from master_thesis.data.ex1.dataset import (
    gen_dataset,
    save_dataset
)


def make_dataset(
    experiment_config: Ex1Config
) -> None:
    dataset = gen_dataset(
        schema=experiment_config.dataset_generation
    )

    save_dataset(
        datataset=dataset,
        path=experiment_config.dataset_path
    )