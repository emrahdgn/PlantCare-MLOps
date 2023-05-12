import ssl
from argparse import Namespace
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models

from config import config

ssl._create_default_https_context = ssl._create_unverified_context

weights_mapping = {
    "efficientnet_b4": "EfficientNet_B4_Weights",
    "efficientnet_b5": "EfficientNet_B5_Weights",
    "efficientnet_b6": "EfficientNet_B6_Weights",
    "efficientnet_b7": "EfficientNet_B7_Weights",
}


class EfficientNetClassifier(nn.Module):
    def __init__(self, args: Namespace, pretrained: bool = True):
        """
        EfficientNet-based image classifier.

        Args:
            args (Namespace): Arguments for model configuration.
            pretrained (bool, optional): Flag indicating whether to use pretrained weights. Defaults to True.

        Raises:
            ValueError: If the specified `train_model_name` is not found in the available models.
        """
        super(EfficientNetClassifier, self).__init__()

        self.train_model_name = args.train_model_name.lower()
        self.weights_name = weights_mapping.get(self.train_model_name, None)
        if self.weights_name is None:
            raise ValueError(
                f"Model {self.train_model_name} not found. Possible options are: 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'"
            )

        self.module = getattr(models, self.train_model_name)
        self.weight = getattr(models, self.weights_name).DEFAULT

        if pretrained:
            self.model = self.module(weights=self.weight)
        else:
            self.model = self.module()

        self.model.classifier[1] = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(self.model.classifier[1].in_features, args.classifier.fc1_output_size)),
                    ("relu1", nn.ReLU()),
                    ("dropout", nn.Dropout(p=args.classifier.dropout_rate)),
                    ("fc2", nn.Linear(args.classifier.fc1_output_size, len(config.ACCEPTED_LABELS))),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EfficientNetClassifier.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def get_model(self) -> nn.Module:
        """
        Get the underlying EfficientNet model.

        Returns:
            nn.Module: EfficientNet model.
        """
        return self.model

    def get_weight(self) -> Any:
        """
        Get the weights of the EfficientNet model.

        Returns:
            Any: EfficientNet model weights.

        """
        return self.weight
