import torch
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from transformers import AutoTokenizer,AutoProcessor
from violet.modeling.modeling_violet import Violet
from violet.modeling.transformer.encoders import VisualEncoder
from violet.modeling.transformer.attention import ScaledDotProductAttention
from typing import List, Union, Dict
from violet.configuration import VioletConfig


class VioletImageCaptioningPipeline:
  """
  Pipeline for image captioning using the Violet model.
  """


  def __init__(self,cfg=VioletConfig):
    """
    Initialize the VioletPipeline.
    Args:
      cfg (VioletConfig): Configuration object containing model parameters.
    """
    self.cfg = cfg
    self.device = self.cfg.DEVICE if self.cfg.DEVICE else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.TOKENIZER_NAME)
    self.processor = AutoProcessor.from_pretrained(self.cfg.PROCESSOR_NAME)
    encoder = VisualEncoder(N=self.cfg.ENCODER_LAYERS,
                            padding_idx=0,
                            attention_module=ScaledDotProductAttention
                            )

    self.model = Violet(
            bos_idx=self.tokenizer.vocab['<|endoftext|>'],
            encoder=encoder,
            n_layer=self.cfg.DECODER_LAYERS,
            tau=self.cfg.TAU,
            device=self.device
        )

    checkpoint = torch.load(self.cfg.CHECKPOINT_DIR, map_location=self.device)
    self.model.load_state_dict(checkpoint['state_dict'], strict=False)
    self.model.to(self.device)
    self.model.eval()

  # .....................................................................

  def prepare_image(self, image: Union[str, np.ndarray,torch.Tensor,Image.Image]) -> Image.Image:
      """
      Prepares an image for feature extraction.

      Args:
          image: Input image (file path, URL, numpy array,tensor, or PIL.Image).

      Returns:
          PIL.Image: The prepared PIL.Image in RGB format.
      """

      if isinstance(image, Image.Image):
        return image.convert("RGB")

      if isinstance(image, str):
          # File path or URL
          if image.startswith("http://") or image.startswith("https://"):
              response = requests.get(image)
              image = Image.open(BytesIO(response.content))
          else:
              image = Image.open(image)
      
      elif isinstance(image, np.ndarray):
          # NumPy array 
          image = Image.fromarray(image)
      elif torch.is_tensor(image):
          # Tensor 
          image = image.permute(1, 2, 0).cpu().numpy()  
          image = Image.fromarray(np.uint8(image))
      else:
          raise ValueError("Unsupported image input type.")
      return image.convert("RGB")



  # .....................................................................

  def generate_features(self, images):
    """
    Extract visual features from the model's encoder.

    Args:
        images (torch.Tensor): Batch of image tensors.

    Returns:
        torch.Tensor: Encoded visual features.
    """
    if not isinstance(images, list):
      images = [images]

    images = list(map(self.prepare_image, images))
  
    images = self.processor(images=images, return_tensors="pt")['pixel_values'].to(self.device)
    with torch.no_grad():
        outputs = self.model.clip(images)
        image_embeds = outputs.image_embeds.unsqueeze(1)  # Add sequence dimension
        features,_ = self.model.encoder(image_embeds)
    return features

  # .....................................................................

  def generate_captions_from_features(self, features):
    """
    Generate captions given pre-extracted visual features.

    Args:
        features (torch.Tensor): Encoded visual features.

    Returns:
        list: Generated captions for each feature set.
    """
    
    with torch.no_grad():
        output,_ = self.model.beam_search(
            visual=features,
            max_len=self.cfg.MAX_LENGTH,
            eos_idx=self.tokenizer.vocab['<|endoftext|>'],
            beam_size=self.cfg.BEAM_SIZE,
            out_size=self.cfg.OUT_SIZE,
            is_feature=True
        )
        captions = [
            [{"caption":self.tokenizer.decode(seq, skip_special_tokens=True)} for seq in output[i]]
            for i in range(output.shape[0])
        ]
       
    return captions


  # .....................................................................

  def generate_captions(self,images):
    """
    Generate captions for a batch of images.

    Args:
        images (torch.Tensor): Batch of image tensors.

    Returns:
        list: Generated captions for each image.
    """
    if not isinstance(images, list):
      images = [images]

    images = list(map(self.prepare_image, images))

    images = self.processor(images=images, return_tensors="pt")['pixel_values'].to(self.device)
    with torch.no_grad():
        output,_ = self.model.beam_search(
            visual=images,
            max_len=self.cfg.MAX_LENGTH,
            eos_idx=self.tokenizer.vocab['<|endoftext|>'],
            beam_size=self.cfg.BEAM_SIZE,
            out_size=self.cfg.OUT_SIZE,
            is_feature=False
        )
        captions = [
            [{"caption":self.tokenizer.decode(seq, skip_special_tokens=True)} for seq in output[i]]
            for i in range(output.shape[0])
        ]
    return captions

  # .....................................................................

  def __call__(self, images):
    """
    Generate captions for a batch of images.

    Args:
        images (torch.Tensor): Batch of image tensors.

    Returns:
        list: Generated captions for each image.
    """
    return self.generate_captions(images)

  # .....................................................................
