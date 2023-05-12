import os
from glob import glob
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import typer
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

app = typer.Typer()


@app.command()
def get_image_paths(images_folder: str, extensions: List[str] = [".jpg", ".png"]) -> List[str]:
    """
    Returns a list of image file paths in the given folder and its subfolders.

    Args:
        images_folder (str): The path to the folder containing the images.
        extensions (list, optional): A list of file extensions to include. Defaults to [".jpg", ".png"].

    Returns:
        list: A list of image file paths.
    """
    image_paths = [y for x in os.walk(images_folder) for ext in extensions for y in glob(os.path.join(x[0], f"*{ext}"))]
    return image_paths


@app.command()
def resize_image(image_path: str, target_size: Tuple[int, int]) -> None:
    """
    Resizes an image to the specified target size using bicubic resampling.

    Args:
        image_path (str): The path to the image file.
        target_size (tuple): The target size of the image as a tuple of two integers (width, height).
    """
    img = Image.open(image_path)
    img_resized = img.resize(target_size, resample=Image.BICUBIC)
    img_resized.save(image_path)


def extract_zip(zip_path: str, destination_path: str) -> None:
    """
    Extracts the contents of a zip file to the specified destination directory.

    Args:
        zip_path (str): The path to the zip file.
        destination_path (str): The path to the destination directory.
    """
    os.makedirs(destination_path, exist_ok=True)
    os.system(f"unzip {zip_path} -d {destination_path}")


def get_data_splits(X: Any, y: Any, train_size: float = 0.7) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Splits the input data into training, validation, and test sets.

    Args:
        X (Any): The input features.
        y (Any): The target variable.
        train_size (float, optional): The proportion of the data to include in the training set. Defaults to 0.7.

    Returns:
        tuple: A tuple containing six elements in the following order:
            - training features (X_train)
            - validation features (X_val)
            - test features (X_test)
            - training target variable (y_train)
            - validation target variable (y_val)
            - test target variable (y_test)
    """
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_, random_state=1)
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, MultiLabelBinarizer]:
    """
    Preprocesses the input DataFrame by transforming the 'labels' column using a MultiLabelBinarizer.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing two elements:
            - The first element is a numpy array containing the transformed labels.
            - The second element is the fitted MultiLabelBinarizer object used to transform the labels.
    """
    df_copy = df.copy()
    df_copy["labels"] = df_copy["labels"].str.split()
    label_encoder = MultiLabelBinarizer().fit(df_copy["labels"])
    return label_encoder.transform(df_copy["labels"]), label_encoder


if __name__ == "__main__":
    app()  # pragma: no cover, live app
