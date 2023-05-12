import copy
import json
import os
import pickle
import random
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn


def set_seeds(SEED: int) -> None:
    """
    Set random seeds for reproducibility.

    This function sets the random seeds for the Python built-in random module, NumPy, and PyTorch,
    including both the CPU and CUDA devices, to ensure consistent results when using random operations.

    Args:
        SEED (int): The seed value to be used for setting the random seeds.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def add_dict_to_dict(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> None:
    """
    Recursively merges two dictionaries.

    This function merges the values from `dict2` into `dict1`. If a key exists in both dictionaries and the values
    associated with the key are dictionaries themselves, the function recursively merges those dictionaries.

    Args:
        dict1 (dict): The target dictionary to merge the values into.
        dict2 (dict): The source dictionary containing values to be merged.

    Raises:
        TypeError: If either `dict1` or `dict2` is not of type `dict`.
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise TypeError("Both dict1 and dict2 must be of type dict.")
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            add_dict_to_dict(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a single-level dictionary.

    This function recursively flattens a nested dictionary `d` into a single-level dictionary, where keys are joined
    using the `sep` separator.

    Args:
        d (dict): The nested dictionary to be flattened.
        parent_key (str): The prefix to be prepended to the keys in the flattened dictionary. (default: "")
        sep (str): The separator used to join the keys in the flattened dictionary. (default: "_")

    Returns:
        dict: The flattened dictionary.

    Examples:
        >>> d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        >>> flatten_dict(d)
        {'a': 1, 'b_c': 2, 'b_d_e': 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_to_string(d: Dict[str, Any]) -> str:
    """
    Convert a dictionary to a string representation.

    This function converts a dictionary `d` into a string representation, where each key-value pair is formatted as
    "key: value" and separated by a new line.

    Args:
        d (dict): The dictionary to be converted to a string.

    Returns:
        str: The string representation of the dictionary.

    Examples:
        >>> d = {"name": "John", "age": 30, "city": "New York"}
        >>> dict_to_string(d)
        'name: John\nage: 30\ncity: New York'
    """
    return "\n".join([f"{key}: {value}" for key, value in d.items()])


def find_max_batch_size(
    model: nn.Module,
    input_img_shape: Tuple[int, int, int] = (3, 224, 224),
    label_num: int = 5,
    device: torch.device = torch.device("cuda"),
    mode: str = "train",
) -> Optional[int]:
    # pragma: no cover, dasd
    """
    Find the maximum batch size that can be used for a given model and input shape.

    This function iteratively increases the batch size until an out-of-memory (OOM) error is encountered, and then
    returns the maximum batch size that did not result in an error.

    Args:
        model (nn.Module): The model to test the batch size for.
        input_img_shape (Tuple[int, int, int]): The shape of the input image in the format (channels, height, width).
            (default: (3, 224, 224))
        label_num (int): The number of labels. (default: 5)
        device (torch.device): The device to use for testing the batch size. (default: torch.device("cuda"))
        mode (str): The mode to run the model in. Can be either "train" or "eval". (default: "train")

    Returns:
        int or None: The maximum batch size that can be used without encountering an OOM error. If no batch size could
            be found (e.g., if the model requires more memory than available), None is returned.

    Raises:
        ValueError: If the batch size cannot be found within a non-zero range.
    """
    batch_size = 1
    max_batch_size = None
    while True:
        try:
            input_shape = tuple([batch_size] + list(input_img_shape))
            inputs = torch.randn(input_shape).to(device)
            labels = torch.randn((tuple([batch_size] + [label_num]))).to(device)
            if mode == "train":
                model.train()
                outputs = model(inputs)
                outputs.sum().backward()
            else:
                with torch.no_grad():
                    model.eval()
                    outputs = model(inputs)
            max_batch_size = batch_size
            batch_size *= 2

            del inputs, labels, outputs
            torch.cuda.empty_cache()

        except Exception:
            batch_size //= 2
            if batch_size == 0:
                raise ValueError("Could not find max batch size")
            else:
                break

    try:
        del inputs, labels, outputs
        torch.cuda.empty_cache()
    except Exception:
        pass

    return max_batch_size


def save_json(content: Any, file_path: str) -> None:
    """
    Save content as JSON to the specified file path.

    This function saves the content as JSON to the specified file path. The file is created if it doesn't exist,
    and the directory structure is created if necessary.

    Args:
        content (Any): The content to be saved as JSON.
        file_path (str): The file path to save the JSON content.

    Returns:
        None

    Raises:
        OSError: If there are any errors creating the directory structure or writing the JSON file.
    """
    os.makedirs(Path(file_path).parent, exist_ok=True)
    with open(file_path, "w") as fp:
        json.dump(content, fp, indent=3)


def read_json(file_path):
    """
    Read JSON content from the specified file path.

    This function reads JSON content from the specified file path and returns it as a Python object.

    Args:
        file_path (str): The file path to read the JSON content from.

    Returns:
        Any: The JSON content as a Python object.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        json.JSONDecodeError: If there is an error decoding the JSON content.
    """
    with open(file_path, "r") as fp:
        content = json.load(fp)
    return content


def set_parameter_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """
    Set the requires_grad attribute of model parameters.

    This function sets the requires_grad attribute of all parameters in the given model to the specified value.

    Args:
        model (nn.Module): The model whose parameters' requires_grad attribute is to be set.
        requires_grad (bool): The value to set for the requires_grad attribute.

    Returns:
        None
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


def make_trainable_only_classifier(model: nn.Module) -> None:
    """
    Make only the classifier parameters trainable in the model.

    This function sets the requires_grad attribute of all parameters in the model except for the parameters in the
    classifier to False. It then sets the requires_grad attribute of the classifier parameters to True.

    Args:
        model (nn.Module): The model whose classifier parameters are to be made trainable.

    Returns:
        None
    """
    set_parameter_requires_grad(model, requires_grad=False)
    for param in model.classifier.parameters():
        param.requires_grad = True


def is_model_on_cuda(model: nn.Module) -> bool:
    """
    Check if the model is currently using CUDA.

    This function checks if the model's parameters are currently located on a CUDA device.

    Args:
        model (nn.Module): The model to check.

    Returns:
        bool: True if the model is on CUDA, False otherwise.
    """
    return next(model.parameters()).is_cuda


def convert_namespace_to_dict(namespace_obj: Namespace) -> Dict[str, Any]:
    """
    Convert a namespace object to a dictionary.

    This function recursively converts a namespace object and its nested namespace objects to a dictionary.

    Args:
        namespace_obj (Namespace): The namespace object to be converted.

    Returns:
        dict: The resulting dictionary with the same structure as the namespace object.

    """
    namespace_obj_copy = copy.deepcopy(namespace_obj)
    namespace_dict = vars(namespace_obj_copy)
    for key, value in namespace_dict.items():
        if isinstance(value, Namespace):
            namespace_dict[key] = convert_namespace_to_dict(value)
    return namespace_dict


def convert_dict_to_namespace(dictionary: Dict[str, Any]) -> Namespace:
    """
    Convert a dictionary to a namespace object.

    This function recursively converts a dictionary and its nested dictionaries to a namespace object.

    Args:
        dictionary (dict): The dictionary to be converted.

    Returns:
        Namespace: The resulting namespace object with the same structure as the dictionary.
    """
    namespace_obj = Namespace(**dictionary)
    for key, value in vars(namespace_obj).items():
        if isinstance(value, dict):
            setattr(namespace_obj, key, convert_dict_to_namespace(value))
    return namespace_obj


def save_object(file_path: str, obj: Any) -> None:
    """
    Save an object to a file using pickle serialization.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (Any): The object to be saved.

    Returns:
        None
    """
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_object(file_path: str) -> Any:
    """
    Load an object from a file using pickle deserialization.

    Args:
        file_path (str): The path to the file containing the object.

    Returns:
        Any: The loaded object.
    """
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Recursively merge two dictionaries.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        dict: The merged dictionary.

    Example:
        >>> dict1 = {"a": 1, "b": {"c": 2}}
        >>> dict2 = {"b": {"d": 3}, "e": 4}
        >>> merged_dict = merge_dicts(dict1, dict2)
    """
    merged = dict1.copy()
    for key, value in dict2.items():
        if isinstance(value, dict):
            merged[key] = merge_dicts(merged.get(key, {}), value)
        else:
            parts = key.split(".")
            current_dict = merged
            for part in parts[:-1]:
                current_dict = current_dict.setdefault(part, {})
            current_dict[parts[-1]] = value
    return merged


def search_folder(folder_name: str, search_directory: str) -> str:
    """
    Search for a folder within a directory and its subdirectories.

    Args:
        folder_name (str): The name of the folder to search for.
        search_directory (str): The directory to search within.

    Returns:
        str: The path of the found folder, or an empty string if the folder is not found.
    """
    folder_path = ""
    for dirpath, dirnames, _ in os.walk(search_directory):
        if folder_name in dirnames:
            folder_path = os.path.join(dirpath, folder_name)

    return folder_path
