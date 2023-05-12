from pathlib import Path
from typing import Dict, Union

import mlflow

from . import hardware, utils


def log_metrics_with_epoch(metrics_dict: Dict[str, Union[float, Dict]], epoch: int, phase: str = "") -> None:
    """
    Log metrics with epoch information using MLflow.

    Args:
        metrics_dict (Dict[str, Union[float, Dict]]): A dictionary containing the metrics to be logged.
            Each metric can be a float value or a nested dictionary of metrics.
        epoch (int): The epoch number.
        phase (str, optional): The phase or stage of the training or evaluation. Defaults to "".
    """
    flattened_metrics = utils.flatten_dict(metrics_dict)
    for metric_name, metric_value in flattened_metrics.items():
        metric_name_with_phase = f"{phase}/{metric_name}" if phase else metric_name
        mlflow.log_metric(metric_name_with_phase, metric_value, step=epoch)


def log_metrics(metrics_dict: Dict[str, Union[float, Dict]], tag: str = "") -> None:
    """
    Log metrics using MLflow.

    Args:
        metrics_dict (Dict[str, Union[float, Dict]]): A dictionary containing the metrics to be logged.
            Each metric can be a float value or a nested dictionary of metrics.
        tag (str, optional): An optional tag or label for the metrics. Defaults to "".
    """
    flattened_metrics = utils.flatten_dict(metrics_dict)
    for metric_name, metric_value in flattened_metrics.items():
        metric_name_with_phase = f"{tag}/{metric_name}" if tag else metric_name
        mlflow.log_metric(metric_name_with_phase, metric_value)


def setup_mlflow(model_registry_dir: str, experiment_name: str) -> None:
    """
    Set up MLflow for tracking experiments and models.

    Args:
        model_registry_dir (str): The directory path where MLflow will store the model registry.
        experiment_name (str): The name of the experiment to be created or used.

    Raises:
        OSError: If an error occurs while creating the model registry directory.
    """
    MODEL_REGISTRY = Path(model_registry_dir)
    Path(MODEL_REGISTRY).mkdir(exist_ok=True)
    mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
    mlflow.set_experiment(experiment_name=experiment_name)


def log_hardware_info() -> None:
    """
    Log hardware information using MLflow tags.

    This function logs the CPU information, GPU information, and OS information as tags in MLflow.

    Raises:
        Exception: If an error occurs while logging the hardware information.
    """
    try:
        cpu_info = hardware.get_cpu_info()
        mlflow.set_tags(cpu_info)

        gpu_info = utils.flatten_dict(hardware.get_gpu_info(), parent_key="Hardware/GPU", sep="/")
        mlflow.set_tags(gpu_info)

        OS_info = hardware.get_OS_info()
        mlflow.set_tags(OS_info)
    except Exception as e:
        raise Exception("Error logging hardware information") from e
