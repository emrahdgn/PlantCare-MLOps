from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image


def predict(images: List[Image.Image], artifacts: Dict[str, Any], batch: bool = False) -> List[str]:
    """
    Performs prediction on the input images using the provided artifacts.

    Args:
        images (List[Image.Image]): The input images to predict labels for.
        artifacts (Dict[str, Any]): The artifacts loaded from the model.
        batch (bool, optional): Whether to process images in batch or individually.
            Defaults to False.

    Returns:
        List[str]: The predicted labels for the input images.

    """
    resize_size = artifacts["test_transforms"].default_transform.resize_size[0]

    if not batch:
        input_images = [artifacts["test_transforms"](img.resize((resize_size, resize_size), resample=Image.BICUBIC)) for img in images]
        input_images = np.stack(input_images, axis=0)
    else:
        input_images = images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_images = torch.tensor(input_images).float().to(device)
    outputs = artifacts["model"].model(input_images)
    preds = (outputs > 0.5).cpu().detach().float()
    pred_labels = artifacts["label_encoder"].inverse_transform(preds)

    return pred_labels
