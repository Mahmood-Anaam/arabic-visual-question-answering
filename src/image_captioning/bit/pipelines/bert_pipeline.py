import torch
import numpy as np
from typing import List, Union, Dict
from PIL import Image
from bit_image_captioning.tokenizers.bert_tokenizer import CaptionTensorizer
from bit_image_captioning.feature_extractors.vinvl import VinVLFeatureExtractor
from pytorch_transformers import BertTokenizer, BertConfig
from bit_image_captioning.modeling.modeling_bert import BertForImageCaptioning


class BiTImageCaptioningPipeline:
    """
    Pipeline for generating image captions using the BiT Image Captioning model.
    """

    def __init__(self, cfg):
        """
        Initializes the pipeline with the specified configuration.

        Args:
            cfg (BiTConfig): Configuration object containing all settings for the pipeline.
        """
        # Load configuration, tokenizer, caption tensorizer, and model
        self.cfg = cfg
        self.config = BertConfig.from_pretrained(cfg.checkpoint)
        self.tokenizer = BertTokenizer.from_pretrained(cfg.checkpoint)
        self.caption_tensorizer = CaptionTensorizer(
            tokenizer=self.tokenizer, is_train=cfg.is_train
        )
        self.feature_extractor = VinVLFeatureExtractor()
        self.model = BertForImageCaptioning.from_pretrained(
            cfg.checkpoint, config=self.config
        )

        # Move model to the configured device
        self.model.to(cfg.device)
        self.model.eval()

        # Automatically set token IDs from the tokenizer
        self._set_token_ids()

    def _set_token_ids(self):
        """
        Sets the token IDs (CLS, SEP, PAD, MASK) in the configuration object.
        """
        cls_token_id, sep_token_id, pad_token_id, mask_token_id = self.tokenizer.convert_tokens_to_ids(
            [
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
                self.tokenizer.mask_token,
            ]
        )
        self.cfg.cls_token_id = cls_token_id
        self.cfg.sep_token_id = sep_token_id
        self.cfg.pad_token_id = pad_token_id
        self.cfg.mask_token_id = mask_token_id

    def _prepare_inputs(self, image: Union[Image.Image, np.ndarray, str]) -> Dict:
        """
        Prepares inputs for the model from a single image.

        Args:
            image: Input image (PIL.Image, NumPy array, or file path).

        Returns:
            Dict: Dictionary of inputs for the model.
        """
        try:
            # Extract image features and object detection labels
            _, image_features, od_labels = self.get_image_features(image)

            # Tensorize inputs using the caption tensorizer
            input_ids, attention_mask, token_type_ids, img_feats, masked_pos = self.caption_tensorizer.tensorize_example(
                text_a=None, img_feat=image_features, text_b=od_labels
            )

            # Prepare inputs as a dictionary
            inputs = {
                "input_ids": input_ids.unsqueeze(0).to(self.cfg.device),  # Batch dim
                "attention_mask": attention_mask.unsqueeze(0).to(self.cfg.device),
                "token_type_ids": token_type_ids.unsqueeze(0).to(self.cfg.device),
                "img_feats": img_feats.unsqueeze(0).to(self.cfg.device),
                "masked_pos": masked_pos.unsqueeze(0).to(self.cfg.device),
                "is_decode": True,
                "do_sample": False,
                "bos_token_id": self.cfg.cls_token_id,
                "pad_token_id": self.cfg.pad_token_id,
                "eos_token_ids": [self.cfg.sep_token_id],
                "mask_token_id": self.cfg.mask_token_id,
                "add_od_labels": self.cfg.add_od_labels,
                "od_labels_start_posid": self.cfg.max_seq_a_length,
                # Beam search hyperparameters
                "max_length": self.cfg.max_gen_length,
                "num_beams": self.cfg.num_beams,
                "temperature": self.cfg.temperature,
                "top_k": self.cfg.top_k,
                "top_p": self.cfg.top_p,
                "repetition_penalty": self.cfg.repetition_penalty,
                "length_penalty": self.cfg.length_penalty,
                "num_return_sequences": self.cfg.num_return_sequences,
                "num_keep_best": self.cfg.num_keep_best,
            }
            return inputs
        except Exception as e:
            raise RuntimeError(f"Failed to prepare inputs for the image: {e}")

    def get_image_features(self, image: Union[Image.Image, np.ndarray, str]):
        """
        Extract image features and object detection labels.

        Args:
            image: Input image (PIL.Image, NumPy array, or file path).

        Returns:
            Tuple: Object detections, image features, and OD labels.
        """
        try:
            # Extract image features using the VinVL feature extractor
            object_detections = self.feature_extractor([image])[0]
            v_feats = np.concatenate(
                (object_detections["features"], object_detections["spatial_features"]),
                axis=1,
            )
            image_features = torch.tensor(v_feats, dtype=torch.float32)

            # Prepare object detection labels if enabled
            if self.cfg.add_od_labels:
                od_labels = " ".join(object_detections["classes"])
            else:
                od_labels = None

            return object_detections, image_features, od_labels
        except Exception as e:
            raise RuntimeError(f"Failed to extract features for the image: {e}")

    def generate_captions(self, images: List[Union[Image.Image, np.ndarray, str]]):
        """
        Generates captions for a list of images.

        Args:
            images: List of images (PIL.Image, NumPy array, or file paths).

        Returns:
            List[List[Dict]]: List of captions with confidence scores for each image.
        """
        results = []
        try:
            for image in images:
                # Prepare inputs for the model
                inputs = self._prepare_inputs(image)

                # Generate captions using the model
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Decode the captions and collect results
                all_caps = outputs[0]
                all_confs = torch.exp(outputs[1])

                captions = []
                for cap, conf in zip(all_caps[0], all_confs[0]):
                    caption = self.tokenizer.decode(
                        cap.tolist(), skip_special_tokens=True
                    )
                    captions.append({"caption": caption, "confidence": conf.item()})

                results.append(captions)

            return results
        except Exception as e:
            raise RuntimeError(f"Failed to generate captions for the images: {e}")


# .......................................................................

