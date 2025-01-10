import torch
from typing import List, Union, Iterable
from PIL import Image
from violet.configuration import VioletConfig
from violet.modeling.modeling_violet import Violet
from violet.modeling.transformer.encoders import VisualEncoder
from violet.modeling.transformer.attention import ScaledDotProductAttention
from transformers import AutoTokenizer, AutoProcessor
import numpy as np
from .base import BaseCaptioner


class VioletCaptioner(BaseCaptioner):
  """
  Implementation of the Violet captioning model.
  """
  def __init__(self, config=VioletConfig):
    """
    Initializes the Violet captioning model with the provided configuration.

    Args:
        config: A configuration object containing model-specific parameters.
        """
    config.device = config.DEVICE
    super().__init__(config)
    self.load_model()
  
  
  def load_model(self):
    """
    Loads the Violet captioning model from the Hugging Face model hub.
    """
    self.tokenizer = AutoTokenizer.from_pretrained(self.config.TOKENIZER_NAME)
    self.processor = AutoProcessor.from_pretrained(self.config.PROCESSOR_NAME)
    encoder = VisualEncoder(N=self.config.ENCODER_LAYERS,
                            padding_idx=0,
                            attention_module=ScaledDotProductAttention
                            )

    self.model = Violet(
            bos_idx=self.tokenizer.vocab['<|endoftext|>'],
            encoder=encoder,
            n_layer=self.config.DECODER_LAYERS,
            tau=self.config.TAU,
            device=self.device
        )

    checkpoint = torch.load(self.config.CHECKPOINT_DIR, map_location=self.device)
    self.model.load_state_dict(checkpoint['state_dict'], strict=False)
    self.model.to(self.device)
    self.model.eval()


  
  def extract_visual_features(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> Iterable:
    """
    Extracts visual features from a batch of images using the Violet model.

    Args:
        images: A list of input images. Can be a list of file paths, URLs, NumPy arrays, or PIL Images.
    
    Returns:
        torch.Tensor: Encoded visual features.
    """

    images = self.prepare_images(images)
    images = self.processor(images=images, return_tensors="pt")['pixel_values'].to(self.device)
    with torch.no_grad():
        outputs = self.model.clip(images)
        image_embeds = outputs.image_embeds.unsqueeze(1)  
        features,_ = self.model.encoder(image_embeds)
    return features



  def generate_captions_from_features(self, features) -> Iterable:
    """
    Generates captions for a batch of visual features using the Violet model.

    Args:
        features (torch.Tensor): Encoded visual features.
    
    Returns:
        A list of lists of dictionaries containing generated captions for each image.
    """
    with torch.no_grad():
      output,_ = self.model.beam_search(
          visual=features,
          max_len=self.config.MAX_LENGTH,
          eos_idx=self.tokenizer.vocab['<|endoftext|>'],
          beam_size=self.config.BEAM_SIZE,
          out_size=self.config.OUT_SIZE,
          is_feature=True
      )

    captions = [
        [{"caption":self.tokenizer.decode(seq, skip_special_tokens=True)} for seq in output[i]]
        for i in range(output.shape[0])
        ]
    return captions

  
  def generate_caption(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> Iterable:
    """
    Generates captions for a batch of images using the Violet model.

    Args:
        images: A list of input images. Can be a list of file paths, URLs, NumPy arrays, or PIL Images.

    Returns:
        A list of lists of dictionaries containing generated captions for each image.
    """
    features = self.extract_visual_features(images)
    return self.generate_captions_from_features(features)
  
  def __call__(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> Iterable:
    """
    Generates captions for a batch of images using the Violet model.

    Args:
        images: A list of input images. Can be a list of file paths, URLs, NumPy arrays, or PIL Images.

    Returns:
        A list of lists of dictionaries containing generated captions for each image.
    """
    return self.generate_caption(images)

# ...................................................................................................
