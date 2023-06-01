from typing import Callable, List, Union

from PIL import Image


class CombinedTrainTransform:
    """
    Applies a combination of default and augmentation transforms to an image.

    Attributes:
        default_transforms (Union[Callable, List[Callable]]): The default transforms to be applied.
        augmentation_transforms (Union[Callable, List[Callable]]): The augmentation transforms to be applied.
    """

    def __init__(self, default_transforms: Union[Callable, List[Callable]], augmentation_transforms: Union[Callable, List[Callable]]):
        """
        Initializes a CombinedTrainTransform object.

        Args:
            default_transforms (Union[Callable, List[Callable]]): The default transforms to be applied.
            augmentation_transforms (Union[Callable, List[Callable]]): The augmentation transforms to be applied.
        """
        self.default_transforms = default_transforms
        self.augmentation_transforms = augmentation_transforms

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Applies the combination of default and augmentation transforms to the input image.

        Args:
            img (Image.Image): The input image.

        Returns:
            Image.Image: The transformed image.
        """
        img = self.augmentation_transforms(img)
        img = self.default_transforms(img)
        return img


class CombinedTestTransform:
    """
    Applies a combination of default transforms to an image for testing.

    Attributes:
        default_transform (Callable): The default transform to be applied.
    """

    def __init__(self, default_transforms: Callable):
        """
        Initializes a CombinedTestTransform object.

        Args:
            default_transform (Callable): The default transform to be applied.
        """
        self.default_transform = default_transforms

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Applies the default transform to the input image.

        Args:
            img (Image.Image): The input image.

        Returns:
            Image.Image: The transformed image.
        """
        img = self.default_transform(img)
        return img
