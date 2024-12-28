from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch


class CaptionGenerator(ABC):
    """Abstract base class for caption generation models, handling various image inputs and generating multiple captions per image."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the CaptionGenerator with the given configuration.

        Args:
            config: A dictionary containing configuration parameters for the model.
        """
        self.config = config
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """
        Loads the pre-trained caption generation model.  This method must be implemented by concrete subclasses.
        """
        pass

    def _prepare_image(self, image: Union[str, np.ndarray, torch.Tensor, Image.Image]) -> Image.Image:
        """
        Prepares a single image for caption generation, converting it to RGB format.  Handles various input types.

        Args:
            image: The input image.  Can be a file path, URL, NumPy array, PyTorch tensor, or PIL Image.

        Returns:
            A PIL Image object in RGB format.

        Raises:
            ValueError: If the input image type is unsupported or if an error occurs during image processing.
            requests.exceptions.RequestException: If there's an error downloading the image from a URL.
        """
        try:
            if isinstance(image, Image.Image):
                return image.convert("RGB")
            elif isinstance(image, str):
                if image.startswith("http"):
                    response = requests.get(image, stream=True)
                    response.raise_for_status()
                    return Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    return Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                return Image.fromarray(image).convert("RGB")
            elif torch.is_tensor(image):
                return Image.fromarray(image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)).convert("RGB")
            else:
                raise ValueError(f"Unsupported image input type: {type(image)}")
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Error downloading image: {e}")
        except Exception as e:
            raise ValueError(f"Error preparing image: {e}")


    @abstractmethod
    def extract_visual_features(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> torch.Tensor:
        """
        Extracts visual features from a batch of images.

        Args:
            images: A list of PIL Image objects or a NumPy array representing a batch of images.

        Returns:
            A PyTorch tensor of shape (num_images, feature_dimension) containing the extracted visual features.
        """
        pass

    @abstractmethod
    def generate_captions_from_features(self, features: torch.Tensor) -> List[List[Dict]]:
        """
        Generates captions from extracted visual features.

        Args:
            features: A PyTorch tensor of shape (num_images, feature_dimension) containing the visual features.

        Returns:
            A list of lists, where each inner list contains dictionaries representing captions for a single image.  Each dictionary should contain at least a 'caption' key with the generated caption string.
        """
        pass


    def generate_captions(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> List[List[Dict]]:
        """
        Generates captions for a batch of images.  Handles various image input types.

        Args:
            images: A list of images or a single image.  Images can be file paths, URLs, NumPy arrays, or PIL Images.

        Returns:
            A list of lists, where each inner list contains dictionaries representing captions for a single image.
            Returns an empty list if there's an error.
        """
        images = [images] if not isinstance(images, list) else images
        try:
            prepared_images = [self._prepare_image(img) for img in images]
            features = self.extract_visual_features(prepared_images)
            return self.generate_captions_from_features(features)
        except Exception as e:
            print(f"Error generating captions: {e}")
            return [[] for _ in images]

    def __call__(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> List[List[Dict]]:
        """
        Allows calling the instance directly as a function.  This is a convenience method.

        Args:
            images: A list of images or a single image.

        Returns:
            The result of generate_captions.
        """
        return self.generate_captions(images)