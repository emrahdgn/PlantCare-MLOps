from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Custom dataset class for image data.

    Args:
        img_names (List[str]): List of image names.
        labels (Union[List[float], np.ndarray]): List or array of labels.
        img_base_folder (str): Base folder path for the images.
        transform (callable, optional): Transform to be applied to the images. Default is None.
    """

    def __init__(self, img_names: List[str], labels: Union[List[float], np.ndarray], img_base_folder: str, transform: callable = None):
        assert len(img_names) == len(labels), "Error: len(img_names) != len(labels)"
        self.img_names = img_names
        self.labels = np.asarray(labels, dtype=np.float32)
        self.img_base_folder = img_base_folder
        self.transform = transform

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.img_names)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: A tuple containing the image and its label.
        """
        img_name = self.img_names[idx]
        img_path = Path(self.img_base_folder, img_name).__str__()
        img = Image.open(img_path).convert("RGB")
        labels = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, labels
