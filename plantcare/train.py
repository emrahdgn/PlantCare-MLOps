import copy
import time
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import evaluate
import model
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import mlflow, utils

from config import config
from config.config import logger
from data import data_functions, data_transforms, dataloader, dataset


def train_and_validate_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    label_name_list: List[str],
    logging_prefix: str = "",
    num_epochs: int = 25,
    trial: Optional[optuna.Trial] = None,
    pruning: bool = False,
) -> Tuple[nn.Module, List[float], Dict[str, Dict[str, float]]]:
    """
    Trains and validates a PyTorch model.

    Args:
        model: The PyTorch model to train and validate.
        dataloaders: A dictionary containing dataloaders for the "train" and "val" phases.
        criterion: The loss criterion for training the model.
        optimizer: The optimizer to use during training.
        label_name_list: A list of label names for evaluation metrics.
        logging_prefix: An optional prefix to use when logging.
        num_epochs: The number of epochs to train the model.
        trial: An optional Optuna trial object for hyperparameter optimization.
        pruning: A flag indicating whether to enable pruning during training.

    Returns:
        A tuple containing the trained model, validation accuracy history, and a dictionary of metrics for each epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.cpu().state_dict())
    model.to(device)
    best_mic_f1 = 0.0
    best_epoch = 0

    all_metrics = {}
    for epoch in range(1, num_epochs + 1):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            all_preds = torch.tensor([])
            all_labels = torch.tensor([])

            running_loss = 0.0
            running_items = 0
            counter = 1

            for inputs, labels in dataloaders[phase]:
                print(f"\rbatch: {counter}/{len(dataloaders[phase])}", end="")
                counter += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    outputs = torch.sigmoid(outputs)
                    loss = criterion(outputs, labels)

                    preds = (outputs > 0.5).cpu().detach().float()

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                all_preds = torch.cat((all_preds, preds.cpu().detach()), dim=0)
                all_labels = torch.cat((all_labels, labels.cpu().detach()), dim=0)

                # statistics
                running_loss += loss.item() * inputs.shape[0]
                running_items += inputs.shape[0]

                del inputs, labels, outputs, loss
                torch.cuda.empty_cache()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            metrics = evaluate.multi_label_classification_metrics(all_labels, all_preds, label_name_list)

            logger.info(f"{logging_prefix} {phase} Loss: {epoch_loss:.5f}  Micro_F1: {metrics['micro_f1']:.5f}")

            if not trial:
                mlflow.log_metrics_with_epoch(metrics_dict={"Loss": epoch_loss, "": metrics}, epoch=epoch, phase=f"{logging_prefix}{phase}")

            # deep copy the model
            if phase == "val":
                all_metrics[f"epoch_{epoch}"] = metrics

                if metrics["micro_f1"] >= best_mic_f1:
                    best_mic_f1 = metrics["micro_f1"]
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.cpu().state_dict())
                    model.to(device)

                if pruning and trial:
                    trial.report(metrics["micro_f1"], epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

    time_elapsed = time.time() - since
    logger.info(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    logger.info(f"Best val Micro F1: {best_mic_f1:4f}")
    all_metrics["best_epoch"] = best_epoch

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, all_metrics


def get_optimizer(model: nn.Module, lr: float) -> optim.Optimizer:
    """
    Returns an optimizer for training the given model.

    Args:
        model: The PyTorch model for which to create the optimizer.
        lr: The learning rate to use for the optimizer.

    Returns:
        The optimizer for training the model.
    """
    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def get_loss_function() -> nn.Module:
    """
    Returns the binary cross-entropy loss function.

    Returns:
        The binary cross-entropy loss function.

    """
    return nn.BCELoss()


def get_transforms(default_transforms: transforms.Compose) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns the train and test data transforms.

    Args:
        default_transforms (transforms.Compose): Default transforms to apply to the data.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: Train and test data transforms.
    """
    # Additional transforms for data augmentation
    augmentation_transforms = transforms.Compose(
        [
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]
    )

    Train_Transforms = data_transforms.CombinedTrainTransform(default_transforms=default_transforms, augmentation_transforms=augmentation_transforms)
    Test_Transforms = data_transforms.CombinedTestTransform(default_transforms=default_transforms)

    return Train_Transforms, Test_Transforms


def get_datasets(
    Train_Transforms: transforms.Compose,
    Test_Transforms: transforms.Compose,
    X_train: List[str],
    y_train: List[int],
    X_val: List[str],
    y_val: List[int],
    X_test: List[str],
    y_test: List[int],
) -> Tuple[dataset.CustomDataset, dataset.CustomDataset, dataset.CustomDataset]:
    """
    Returns the train, validation, and test datasets.

    Args:
        Train_Transforms (transforms.Compose): Transforms to apply to the training dataset.
        Test_Transforms (transforms.Compose): Transforms to apply to the validation and test datasets.
        X_train (List[str]): List of training image names.
        y_train (List[int]): List of training labels.
        X_val (List[str]): List of validation image names.
        y_val (List[int]): List of validation labels.
        X_test (List[str]): List of test image names.
        y_test (List[int]): List of test labels.

    Returns:
        Tuple[dataset.CustomDataset, dataset.CustomDataset, dataset.CustomDataset]: Train, validation, and test datasets.
    """
    train_dataset = dataset.CustomDataset(img_names=X_train, labels=y_train, img_base_folder=f"{config.DATA_DIR}/images", transform=Train_Transforms)
    val_dataset = dataset.CustomDataset(img_names=X_val, labels=y_val, img_base_folder=f"{config.DATA_DIR}/images", transform=Test_Transforms)
    test_dataset = dataset.CustomDataset(img_names=X_test, labels=y_test, img_base_folder=f"{config.DATA_DIR}/images", transform=Test_Transforms)

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    train_dataset: dataset.CustomDataset,
    val_dataset: dataset.CustomDataset,
    test_dataset: dataset.CustomDataset,
    train_batch_size: int,
    test_batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns the train, validation, and test dataloaders.

    Args:
        train_dataset (dataset.CustomDataset): Training dataset.
        val_dataset (dataset.CustomDataset): Validation dataset.
        test_dataset (dataset.CustomDataset): Test dataset.
        train_batch_size (int): Batch size for the training dataloader.
        test_batch_size (int): Batch size for the validation and test dataloaders.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders.
    """
    train_dataloader, val_dataloader, test_dataloader = dataloader.create_train_val_dataloders(
        train_dataset, val_dataset, test_dataset, train_batch_size, test_batch_size
    )

    return train_dataloader, val_dataloader, test_dataloader


def find_max_batch_sizes(model: nn.Module, input_size: int, label_num: int, device: torch.device) -> Tuple[int, int, int]:
    """
    Finds the maximum batch sizes for training and testing with the given model.

    Args:
        model: The model to evaluate.
        input_size (int): The input size of the model (assumed to be square).
        label_num (int): The number of labels.
        device: The device to use for evaluation (e.g., "cuda" or "cpu").

    Returns:
        Tuple[int, int, int]: Maximum batch sizes for partial training, full training, and testing.

    Raises:
        TypeError: If the device is not torch.device type.

    Note:
        If the device is not CUDA, default values of (2, 2, 2) are returned.
    """
    if device != torch.device("cuda"):
        return 2, 2, 2

    utils.set_parameter_requires_grad(model, True)
    full_train_max_batch_size = utils.find_max_batch_size(
        model, input_img_shape=(3, input_size, input_size), label_num=label_num, device=device, mode="train"
    )

    test_max_batch_size = utils.find_max_batch_size(
        model, input_img_shape=(3, input_size, input_size), label_num=label_num, device=device, mode="test"
    )

    utils.make_trainable_only_classifier(model)
    partial_train_max_batch_size = utils.find_max_batch_size(
        model, input_img_shape=(3, input_size, input_size), label_num=label_num, device=device, mode="train"
    )

    torch.cuda.empty_cache()
    return partial_train_max_batch_size, full_train_max_batch_size, test_max_batch_size


def objective(args: Namespace, trial: optuna.trial._trial.Trial) -> float:
    """
    Objective function for optimization trials.

    Args:
        args (Namespace): Namespace object containing the arguments.
        trial (optuna.trial._trial.Trial): Optuna Trial object for hyperparameter optimization.

    Returns:
        float: Micro F1 performance metric.

    """
    # Parameters to tune
    args.train_model_name = trial.suggest_categorical("train_model_name", ["efficientnet_b5", "efficientnet_b6", "efficientnet_b7"])
    args.finetuning_part_1.num_epoch = trial.suggest_categorical("finetuning_part_1.num_epoch", [1, 2, 3])
    args.finetuning_part_2.num_epoch = trial.suggest_categorical("finetuning_part_2.num_epoch", [2, 3, 4])
    args.finetuning_part_1.lr = trial.suggest_float("finetuning_part_1.lr", 1e-4, 1e-2, log=True)
    args.finetuning_part_2.lr = trial.suggest_float("finetuning_part_2.lr", 1e-5, 5e-3, log=True)

    df = pd.read_csv("data/labels.csv")
    labels, label_encoder = data_functions.preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = data_functions.get_data_splits(X=df.image.to_numpy(), y=labels, train_size=0.7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Classifier_Model = model.EfficientNetClassifier(args=args)
    model_instance = Classifier_Model.get_model()
    weight = Classifier_Model.get_weight()

    model_instance.to(device)

    input_size = weight.transforms.keywords.get("resize_size")
    partial_train_max_batch_size, full_train_max_batch_size, test_max_batch_size = find_max_batch_sizes(
        model=model_instance, input_size=input_size, label_num=len(label_encoder.classes_), device=device
    )

    Train_transforms, Test_transforms = get_transforms(default_transforms=weight.transforms())

    train_dataset, val_dataset, test_dataset = get_datasets(Train_transforms, Test_transforms, X_train, y_train, X_val, y_val, X_test, y_test)
    partial_train_dataloader, partial_val_dataloader, _ = get_dataloaders(
        train_dataset, val_dataset, test_dataset, partial_train_max_batch_size, test_max_batch_size
    )
    full_train_dataloader, full_val_dataloader, _ = get_dataloaders(
        train_dataset, val_dataset, test_dataset, full_train_max_batch_size, test_max_batch_size
    )

    criterion = get_loss_function()
    utils.make_trainable_only_classifier(model_instance)
    optimizer = get_optimizer(model=model_instance, lr=args.finetuning_part_1.lr)

    # Train & evaluate
    model_instance, _, _ = train_and_validate_model(
        model=model_instance,
        dataloaders={"train": partial_train_dataloader, "val": partial_val_dataloader},
        criterion=criterion,
        optimizer=optimizer,
        label_name_list=label_encoder.classes_,
        logging_prefix="Classfier",
        num_epochs=args.finetuning_part_1.num_epoch,
        trial=trial,
        pruning=False,
    )

    model_instance.to(device)

    utils.set_parameter_requires_grad(model=model_instance, requires_grad=True)
    optimizer = get_optimizer(model=model_instance, lr=args.finetuning_part_2.lr)

    model_instance, _, performances = train_and_validate_model(
        model=model_instance,
        dataloaders={"train": full_train_dataloader, "val": full_val_dataloader},
        criterion=criterion,
        optimizer=optimizer,
        label_name_list=label_encoder.classes_,
        logging_prefix="Whole",
        num_epochs=args.finetuning_part_2.num_epoch,
        trial=trial,
        pruning=True,
    )

    best_epoch = performances["best_epoch"]
    performance = performances[f"epoch_{best_epoch}"]

    performance_flattened = utils.flatten_dict(performance)
    for key, value in performance_flattened.items():
        trial.set_user_attr(key, value)

    logger.info(f"Trial {trial.number + 1} is finished")

    return performance["micro_f1"]
