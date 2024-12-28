from aravqa.modules.captioning.base import CaptionGenerator
from typing import List, Dict, Union
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from violet.modeling.modeling_violet import Violet
from violet.modeling.transformer.encoders import VisualEncoder
from violet.modeling.transformer.attention import ScaledDotProductAttention
from violet.configuration import VioletConfig


class VioletCaptioner(CaptionGenerator):
    """
    Generates captions for images using the Violet model.
    """

    def __init__(self, config: VioletConfig):
        """
        Initializes the VioletCaptioner.

        Args:
            config: A VioletConfig object containing model parameters.
        """
        super().__init__(config)
        self.cfg = config
        self.device = self.cfg.DEVICE if self.cfg.DEVICE else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.TOKENIZER_NAME)
        self.processor = AutoProcessor.from_pretrained(self.cfg.PROCESSOR_NAME)
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()


    def _load_model(self) -> Violet:
        """
        Loads the pre-trained Violet model.

        Returns:
            A Violet model instance.

        Raises:
            FileNotFoundError: If the checkpoint file is not found.
            Exception: If any other error occurs during model loading.
        """
        try:
            encoder = VisualEncoder(N=self.cfg.ENCODER_LAYERS, padding_idx=0, attention_module=ScaledDotProductAttention)
            model = Violet(
                bos_idx=self.tokenizer.vocab['<|endoftext|>'],
                encoder=encoder,
                n_layer=self.cfg.DECODER_LAYERS,
                tau=self.cfg.TAU,
                device=self.device
            )
            checkpoint = torch.load(self.cfg.CHECKPOINT_DIR, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {self.cfg.CHECKPOINT_DIR}")
        except Exception as e:
            raise Exception(f"Error loading Violet model: {e}")


    

    def extract_visual_features(self,images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> torch.Tensor:
        """
        Extracts visual features from a batch of images using the Violet model's CLIP encoder.

        Args:
            images: A list of PIL Image objects or a NumPy array representing a batch of images.

        Returns:
            A PyTorch tensor of shape (num_images, feature_dimension) containing the extracted visual features.

        Raises:
            TypeError: if input images are not PIL images or NumPy array.
            Exception: if any error occurs during feature extraction.
        """
        try:
            
            images = [images] if not isinstance(images, list) else images
            images = list(map(self._prepare_image, images))
            images = self.processor(images=images, return_tensors="pt")['pixel_values'].to(self.device)
            with torch.no_grad():
                outputs = self.model.clip(images)
                image_embeds = outputs.image_embeds.unsqueeze(1)
                features, _ = self.model.encoder(image_embeds)
            return features
        except Exception as e:
            raise Exception(f"Error extracting visual features: {e}")


    def generate_captions_from_features(self, features: torch.Tensor) -> List[List[Dict]]:
        """
        Generates captions from pre-extracted visual features using beam search.

        Args:
            features: A PyTorch tensor of shape (num_images, feature_dimension) containing the visual features.

        Returns:
            A list of lists, where each inner list contains dictionaries representing captions for a single image.
            Each dictionary contains a 'caption' key with the generated caption string.

        Raises:
            Exception: If any error occurs during caption generation.
        """
        try:
            with torch.no_grad():
                output, _ = self.model.beam_search(
                    visual=features,
                    max_len=self.cfg.MAX_LENGTH,
                    eos_idx=self.tokenizer.vocab['<|endoftext|>'],
                    beam_size=self.cfg.BEAM_SIZE,
                    out_size=self.cfg.OUT_SIZE,
                    is_feature=True
                )
                captions = [
                    [{"caption": self.tokenizer.decode(seq, skip_special_tokens=True)} for seq in output[i]]
                    for i in range(output.shape[0])
                ]
            return captions
        except Exception as e:
            raise Exception(f"Error generating captions from features: {e}")


    def generate_captions(self,images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> List[List[Dict]]:
        """
        Generates captions for a batch of images. Handles various image input types.

        Args:
            images: A list of images or a single image. Images can be file paths, URLs, NumPy arrays, or PIL Images.

        Returns:
            A list of lists, where each inner list contains dictionaries representing captions for a single image.
            Each dictionary contains a 'caption' key with the generated caption string.  Returns an empty list if there's an error.
        """
        images = [images] if not isinstance(images, list) else images

        try:
            images = list(map(self._prepare_image, images))
            images = self.processor(images=images, return_tensors="pt")['pixel_values'].to(self.device)
            prepared_images = [self._prepare_image(img) for img in images]
            return self.generate_captions_from_features(self.extract_visual_features(prepared_images))
        except Exception as e:
            print(f"Error generating captions: {e}")
            return [[] for _ in images]

    def __call__(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> List[List[Dict]]:
        """
        Allows calling the instance directly as a function. This is a convenience method.

        Args:
            images: A list of images or a single image.

        Returns:
            The result of generate_captions.
        """
        return self.generate_captions(images)