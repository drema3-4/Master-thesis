from master_thesis.schemas.ex1.schemas import (
    Ex1Config
)
from master_thesis.data.ex1.dataset import (
    calibration_dataset_parameters_by_monte_carlo,
    choose_dataset_params,
    gen_dataset,
    save_dataset
)


def make_dataset(
    experiment_config: Ex1Config
) -> None:
    calibration_dataset = calibration_dataset_parameters_by_monte_carlo(
        experiment_config.dataset_calibration_params
    )
    save_dataset(
        datataset=calibration_dataset,
        path=experiment_config.calibration_dataset_path
    )

    choose_dataset_params_ = choose_dataset_params(
        calibration_dataset_path=experiment_config.calibration_dataset_path,
        target_evidence_strength=experiment_config.target_evidence_strength
    )
    save_dataset(
        datataset=choose_dataset_params_,
        path=experiment_config.choose_dataset_params_path
    )

    dataset = gen_dataset(
        choose_dataset_params_path=experiment_config.choose_dataset_params_path
    )
    save_dataset(
        datataset=dataset,
        path=experiment_config.dataset_path
    )