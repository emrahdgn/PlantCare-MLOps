import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from pathlib import Path
from typing import Any, Dict, Union

import mlflow
import model
import numpy as np
import optuna
import pandas as pd
import predict
import torch
import train
import typer
from optuna.integration.mlflow import MLflowCallback
from PIL import Image
from utils import mlflow as mlflow_utils
from utils import utils

from config import config
from config.config import logger
from data import data_functions

app = typer.Typer()

utils.set_seeds(SEED=1)


@app.command()
def train_model(
    args_fp: str = "config/args.json",
    experiment_name: str = "baselines",
    run_name: str = "efficient_Net",
    test_run: bool = False,
) -> None:
    """
    Train and validate a model using the specified configuration.

    Args:
        args_fp (str): The file path to the JSON file containing the configuration arguments.
        experiment_name (str): The name of the MLflow experiment.
        run_name (str): The name of the MLflow run.
        test_run (bool): Whether it is a test run or an actual run.

    Returns:
        None
    """
    args = utils.convert_dict_to_namespace(utils.read_json(file_path=args_fp))

    df = pd.read_csv("data/labels.csv")
    labels, label_encoder = data_functions.preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = data_functions.get_data_splits(X=df.image.to_numpy(), y=labels, train_size=0.7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Classifier_Model = model.EfficientNetClassifier(args=args)
    model_instance = Classifier_Model.get_model()
    weight = Classifier_Model.get_weight()

    model_instance.to(device)

    input_size = weight.transforms.keywords.get("resize_size")
    partial_train_max_batch_size, full_train_max_batch_size, test_max_batch_size = train.find_max_batch_sizes(
        model=model_instance, input_size=input_size, label_num=len(label_encoder.classes_), device=device
    )

    Train_transforms, Test_transforms = train.get_transforms(default_transforms=weight.transforms())

    train_dataset, val_dataset, test_dataset = train.get_datasets(Train_transforms, Test_transforms, X_train, y_train, X_val, y_val, X_test, y_test)
    partial_train_dataloader, partial_val_dataloader, _ = train.get_dataloaders(
        train_dataset, val_dataset, test_dataset, partial_train_max_batch_size, test_max_batch_size
    )
    full_train_dataloader, full_val_dataloader, _ = train.get_dataloaders(
        train_dataset, val_dataset, test_dataset, full_train_max_batch_size, test_max_batch_size
    )

    criterion = train.get_loss_function()
    utils.make_trainable_only_classifier(model_instance)
    optimizer = train.get_optimizer(model=model_instance, lr=args.finetuning_part_1.lr)

    mlflow_utils.setup_mlflow(model_registry_dir=config.MLFLOW_STORE / "Experiments", experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        experiment_id = mlflow.active_run().info.experiment_id

        mlflow_utils.log_hardware_info()
        mlflow.log_params(utils.flatten_dict(utils.convert_namespace_to_dict(args), sep="/"))
        mlflow.log_params(
            utils.flatten_dict(
                {
                    "sample number": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
                    "batch size": {
                        "classifier_only": partial_train_max_batch_size,
                        "full_model": full_train_max_batch_size,
                        "test": test_max_batch_size,
                    },
                },
                sep="/",
            )
        )

        model_instance, _, _ = train.train_and_validate_model(
            model=model_instance,
            dataloaders={"train": partial_train_dataloader, "val": partial_val_dataloader},
            criterion=criterion,
            optimizer=optimizer,
            label_name_list=label_encoder.classes_,
            logging_prefix="Classfier",
            num_epochs=args.finetuning_part_1.num_epoch,
        )

        model_instance.to(device)

        utils.set_parameter_requires_grad(model=model_instance, requires_grad=True)
        optimizer = train.get_optimizer(model=model_instance, lr=args.finetuning_part_2.lr)

        model_instance, _, full_metrics = train.train_and_validate_model(
            model=model_instance,
            dataloaders={"train": full_train_dataloader, "val": full_val_dataloader},
            criterion=criterion,
            optimizer=optimizer,
            label_name_list=label_encoder.classes_,
            logging_prefix="Full",
            num_epochs=args.finetuning_part_2.num_epoch,
        )

        Artifacts_Dir = Path(config.MODEL_REGISTRY, "Experiments", experiment_id, run_id, "artifacts")
        os.makedirs(Artifacts_Dir, exist_ok=True)

        torch.save(model_instance, Path(Artifacts_Dir, "model.pth"))
        torch.save(optimizer, Path(Artifacts_Dir, "optimizer.pth"))

        utils.save_json(utils.convert_namespace_to_dict(args), Path(Artifacts_Dir, "args.json"))
        utils.save_object(file_path=Path(Artifacts_Dir, "Train_Transforms.pkl"), object=Train_transforms)
        utils.save_object(file_path=Path(Artifacts_Dir, "Test_Transforms.pkl"), object=Test_transforms)
        utils.save_object(file_path=Path(Artifacts_Dir, "Label_Encoder.pkl"), object=label_encoder)
        utils.save_json(content=full_metrics, file_path=Path(Artifacts_Dir, "performance.json"))

        if not test_run:  # pragma: no cover, actual run
            with open(Path(config.CONFIG_DIR, "run_id.txt"), "w") as f:
                f.write(run_id)


@app.command()
def optimize(args_fp: str = "config/args.json", study_name: str = "optimization", num_trials: int = 30) -> None:
    """
    Optimize hyperparameters using Optuna.

    Args:
        args_fp (str): The file path to the JSON file containing the configuration arguments.
        study_name (str): The name of the Optuna study.
        num_trials (int): The number of optimization trials.

    Returns:
        None
    """
    args = utils.convert_dict_to_namespace(utils.read_json(file_path=args_fp))

    Artifacts_Dir = Path(config.MODEL_REGISTRY, "Optimization_Tasks", study_name)
    Artifacts_Dir.mkdir(parents=True, exist_ok=True)

    Mlflow_save_path = config.MLFLOW_STORE / "Optimization_Tasks"
    Mlflow_save_path.mkdir(parents=True, exist_ok=True)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=f"file://{str(Mlflow_save_path)}", metric_name="micro_f1")
    study.optimize(lambda trial: train.objective(args, trial), n_trials=num_trials, callbacks=[mlflow_callback])
    utils.save_object(str(Artifacts_Dir / "study.pkl"), study)

    trials_df = study.trials_dataframe()
    trials_df = trials_df.rename(columns=lambda x: x.replace("user_attrs_", ""))
    trials_df = trials_df.sort_values(["micro_f1"], ascending=False)  # sort by metric
    trials_df.to_csv(str(Artifacts_Dir / "trials.csv"), index=False)

    args_dict = utils.read_json(file_path=args_fp)
    new_args_dict = utils.merge_dicts(args_dict, study.best_trial.params)
    utils.save_json(new_args_dict, args_fp)
    logger.info(f"Best value (f1): {study.best_trial.value}")
    logger.info(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")


@app.command()
def load_artifacts(run_id: str = None, mode: str = "train") -> Dict[str, Any]:
    """
    Load artifacts from a specified run_id.

    Args:
        run_id (str, optional): The ID of the run. If not provided, it will be read from the 'run_id.txt' file in the CONFIG_DIR directory.
        mode (str, optional): The mode of operation. Valid values are 'train' and 'test'. Defaults to 'train'.

    Returns:
        dict: A dictionary containing the loaded artifacts.

    Raises:
        FileNotFoundError: If no matching 'run_id' is found in any subdirectory under 'outputs/model/Experiments' in test mode.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == "test":
        run_id_dir = utils.search_folder(folder_name=run_id, search_directory="outputs/model/Experiments")
        if run_id_dir == "":
            raise FileNotFoundError(
                "No matching run_id found in any subdirectory under 'outputs/model/Experiments'. Please make sure you have specified a valid run_id and that the directory structure is correct."
            )

        artifacts_dir = Path(run_id_dir, "artifacts")
    else:
        # Locate specifics artifacts directory
        mlflow.set_tracking_uri("file://" + str(Path(config.MLFLOW_STORE, "Experiments").absolute()))
        experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
        artifacts_dir = Path(config.MODEL_REGISTRY, "Experiments", experiment_id, run_id, "artifacts")

        if not os.path.exists(artifacts_dir):
            return {f"artifacts for {run_id} not exist in local file system"}

    # Load objects from run
    args = utils.convert_dict_to_namespace(utils.read_json(Path(artifacts_dir, "args.json")))
    test_transforms = utils.load_object(Path(artifacts_dir, "Test_Transforms.pkl"))
    label_encoder = utils.load_object(Path(artifacts_dir, "Label_Encoder.pkl"))

    Classifier_Model = model.EfficientNetClassifier(args=args, pretrained=False)
    Classifier_Model.model = torch.load(Path(artifacts_dir, "model.pth"), map_location=device)

    performance = utils.read_json(file_path=Path(artifacts_dir, "performance.json"))
    best_epoch_num = performance["best_epoch"]

    if mode == "train":
        train_transforms = utils.load_object(Path(artifacts_dir, "Train_Transforms.pkl"))
        optimizer = torch.load(Path(artifacts_dir, "optimizer.pth"))

        return {
            "args": args,
            "train_transforms": train_transforms,
            "test_transforms": test_transforms,
            "label_encoder": label_encoder,
            "model": Classifier_Model,
            "performance": performance,
            "optimizer": optimizer,
        }

    performance = performance[f"epoch_{best_epoch_num}"]
    Classifier_Model.model.eval()

    return {"args": args, "test_transforms": test_transforms, "label_encoder": label_encoder, "model": Classifier_Model, "performance": performance}


@app.command()
def predict_label(image: Union[Image.Image, np.ndarray, str], run_id: str = None) -> Dict[str, Any]:
    """
    Predict the label for a given image using the specified run_id.

    Args:
        image (Union[Image.Image, np.ndarray, str]): The input image to predict the label for. It can be a PIL.Image object, a numpy array, or a file path.
        run_id (str, optional): The ID of the run to load the artifacts from. If not provided, it will be read from the 'run_id.txt' file in the CONFIG_DIR directory.

    Returns:
        dict: A dictionary containing the prediction result.

    Raises:
        ValueError: If the image type is not supported.

    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    artifacts = load_artifacts(run_id=run_id)

    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError("Unsupported image type. Please provide a PIL.Image object, a numpy array, or a file path.")

    predictions = predict.predict(image, artifacts=artifacts)

    return predictions


if __name__ == "__main__":
    app() # pragma: no cover, live app
