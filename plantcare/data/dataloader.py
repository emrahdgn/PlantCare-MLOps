import os
from typing import Tuple

from torch.utils.data import DataLoader, Dataset


def create_train_val_dataloders(
    train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, train_batch_size: int, test_batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates data loaders for the training, validation, and test datasets.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The test dataset.
        train_batch_size (int): The batch size for the training data loader.
        test_batch_size (int): The batch size for the validation and test data loaders.

    Returns:
        tuple: A tuple containing three DataLoader objects:
            - The first DataLoader is for the training dataset and has a batch size of `train_batch_size`.
            - The second DataLoader is for the validation dataset and has a batch size of `test_batch_size`.
            - The third DataLoader is for the test dataset and also has a batch size of `test_batch_size`.
    """
    num_workers = min(16, os.cpu_count())

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader
