from abc import ABC, abstractmethod
from typing import List, Union, Iterable
from PIL import Image
import numpy as np
import torch
from io import BytesIO
import requests

class BaseCaptioner(ABC):
  """
  Abstract Base Class for image captioning models.
  This class defines a unified interface for captioning models, 
  allowing single and batch image processing with any input format.
  """

  def __init__(self, config):
    """
    Initializes the captioning model with the provided configuration.
    
    Args:
        config: A configuration object containing model-specific parameters.
    """
    self.config = config
    self.device = config.device if config.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def _prepare_image(self, image: Union[str, np.ndarray, torch.Tensor, Image.Image]) -> Image.Image:
    """
    Prepares a single image for caption generation, converting it to RGB format. Handles various input types.

    Args:
        image: The input image. Can be a file path, URL, NumPy array, PyTorch tensor, or PIL Image.
    
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
  

  def prepare_images(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> List[Image.Image]:
    """
    Prepares a batch of images for caption generation by converting them to PIL Images in RGB format.

    Args:
        images: A list of input images. Can be a list of file paths, URLs, NumPy arrays, or PIL Images.

    Returns:
        A list of PIL Image objects in RGB format.
    """

    if not isinstance(images, list):
      images = [images]

    return [self._prepare_image(image) for image in images]
  

  @abstractmethod
  def load_model(self):
    """
    Abstract method to load the model weights.
    """
    pass


  
  @abstractmethod
  def extract_visual_features(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> Iterable:
    """
    Abstract method to extract visual features from a batch of images.

    Args:
        images: A list of input images. Can be a list of file paths, URLs, NumPy arrays, or PIL Images.
    
    Returns:
        Any type of iterable object containing visual features for each image.
    """
    pass
  

  @abstractmethod
  def generate_captions_from_features(self, features: Iterable) -> Iterable:
    """
    Abstract method to generate captions from visual features.

    Args:
        features: any type of iterable object containing visual features for each image.
    
    Returns:
        Any type of iterable object containing generated captions for each image.
    """
    pass

  @abstractmethod
  def generate_caption(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> Iterable:
    """
    Abstract method to generate captions for a batch of images.

    Args:
        images: A list of input images. Can be a list of file paths, URLs, NumPy arrays, or PIL Images.
    
    Returns:
        Any type of iterable object containing generated captions for each image
    """
    pass
  

  def __call__(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> Iterable:
    """
    Generates captions for a batch of images.

    Args:
        images: A list of input images. Can be a list of file paths, URLs, NumPy arrays, or PIL Images.
    
    Returns:
        Any type of iterable object containing generated captions for each image.
    """
    return self.generate_caption(images)
    
# ...................................................................................................
