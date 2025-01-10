import torch
import numpy as np
from typing import List, Union,Iterable
from PIL import Image
from huggingface_hub import snapshot_download
from vinvl_bert.tokenizers.bert_tokenizer import CaptionTensorizer
from vinvl_bert.feature_extractors import VinVLFeatureExtractor
from pytorch_transformers import BertTokenizer, BertConfig
from vinvl_bert.modeling.modeling_bert import BertForImageCaptioning
from vinvl_bert.configs import VinVLBertConfig
from .base import BaseCaptioner


class BiTCaptioner(BaseCaptioner):
  """
  Implementation of the BiT (VinVL-BERT) captioning model.
  """

  def __init__(self, config=VinVLBertConfig):
    """
    Initializes the BiT captioning model with the provided configuration.

    Args:
        config: A configuration object containing model-specific parameters.
    """
    super().__init__(config)
    self.load_model()
  
  
  def load_model(self):
    """
    Loads the BiT captioning model from the Hugging Face model hub.
    """
    # dowbload the model from huggingface
    checkpoint = snapshot_download(repo_id=self.config.model_id,repo_type="model")

    # Load configuration, tokenizer, caption tensorizer, and model
       
    self.bertconfig = BertConfig.from_pretrained(checkpoint)
    self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
    self.caption_tensorizer = CaptionTensorizer(
        tokenizer=self.tokenizer, is_train=self.config.is_train
    )
    self.feature_extractor = VinVLFeatureExtractor(device=self.config.device)
    self.model = BertForImageCaptioning.from_pretrained(
        checkpoint, config=self.bertconfig
    )
    
    self.model.to(self.config.device)
    self.model.eval()

    tokens = [self.tokenizer.cls_token, self.tokenizer.sep_token,
              self.tokenizer.pad_token, self.tokenizer.mask_token
             ]
    (self.cls_token_id,
     self.sep_token_id,
     self.pad_token_id, 
     self.mask_token_id )= self.tokenizer.convert_tokens_to_ids(tokens)


    self.input_parms ={
            "is_decode": True,
            "do_sample": False,
            "bos_token_id": self.cls_token_id,
            "pad_token_id": self.pad_token_id,
            "eos_token_ids": [self.sep_token_id],
            "mask_token_id": self.mask_token_id,
            "add_od_labels": self.config.add_od_labels,
            "od_labels_start_posid": self.config.max_seq_a_length,
            # Beam search hyperparameters
            "max_length": self.config.max_gen_length,
            "num_beams": self.config.num_beams,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "length_penalty": self.config.length_penalty,
            "num_return_sequences": self.config.num_return_sequences,
            "num_keep_best": self.config.num_keep_best,
        }
  

  def extract_visual_features(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> Iterable:
    """
    Extracts visual features from a batch of images using the BiT model.

    Args:
        images: A list of input images. Can be a list of file paths, URLs, NumPy arrays, or PIL Images.
    
    Returns:
        A list of dictionaries containing visual features for each image.
    """
    features = self.feature_extractor(images)
    return features
  
  def prepare_inputs(self,features):
    """
    Prepares the input for the BiT model.
    Args:
        features: A list of dictionaries containing visual features for each image.
    Returns:
        A dictionary containing the input tensors for the BiT model.
    """
    inputs = []
    for feature in features:
      image_features, od_labels = feature["img_feats"],feature["od_labels"]
      # Tensorize inputs using the caption tensorizer
      (input_ids,
      attention_mask,
      token_type_ids,
      img_feats, 
      masked_pos) = self.caption_tensorizer.tensorize_example(text_a=None,
                                                              img_feat=image_features,
                                                              text_b=od_labels
                                                              )

      input = {
              "input_ids": input_ids.unsqueeze(0).to(self.config.device),  
              "attention_mask": attention_mask.unsqueeze(0).to(self.config.device),
              "token_type_ids": token_type_ids.unsqueeze(0).to(self.config.device),
              "img_feats": img_feats.unsqueeze(0).to(self.config.device),
              "masked_pos": masked_pos.unsqueeze(0).to(self.config.device),
          }
      
      input.update(self.input_parms)
      inputs.append(input)

    return inputs


  def generate_captions_from_features(self, features) -> Iterable:
    """
    Generates captions for a batch of visual features using the BiT model.

    Args:
        features: A list of dictionaries containing visual features for each image.
    
    Returns:
        A list of lists of dictionaries containing generated captions for each image.
    """
    captions = []
    inputs = self.prepare_inputs(features)
    for input in inputs:
      # Generate captions using the model
      with torch.no_grad():
        outputs = self.model(**input)
      # Decode the captions and collect results
      all_caps = outputs[0]
      all_confs = torch.exp(outputs[1])
      caps = []
      for cap, conf in zip(all_caps[0], all_confs[0]):
        caption = self.tokenizer.decode(cap.tolist(), skip_special_tokens=True)
        caps.append({"caption": caption, "confidence": conf.item()})
      captions.append(caps)
    return captions


  def generate_caption(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> Iterable:
    """
    Generates captions for a batch of images using the BiT model.

    Args:
        images: A list of input images. Can be a list of file paths, URLs, NumPy arrays, or PIL Images.

    Returns:
        A list of lists of dictionaries containing generated captions for each image.
    """
    features = self.extract_visual_features(images)
    return self.generate_captions_from_features(features)
  
  def __call__(self, images: Union[List[Union[str, np.ndarray, Image.Image]], np.ndarray, str, Image.Image]) -> Iterable:
    """
    Generates captions for a batch of images using the BiT model.

    Args:
        images: A list of input images. Can be a list of file paths, URLs, NumPy arrays, or PIL Images.
    
    Returns:
        A list of lists of dictionaries containing generated captions for each image.
    """
    return self.generate_caption(images)
 
# ...................................................................................................