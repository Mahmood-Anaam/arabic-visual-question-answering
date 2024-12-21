import torch
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from io import BytesIO

from scene_graph_benchmark.wrappers.wrappers import VinVLVisualBackbone
from scene_graph_benchmark.wrappers.utils import encode_spatial_features

class VinVLFeatureExtractor(VinVLVisualBackbone):
    """
    VinVL Feature Extractor for extracting visual features from images.
    Supports various input types (file path, URL, PIL.Image, numpy array, and torch.Tensor).
    """
    
    def __init__(self, config_file=None, opts=None,add_od_labels=True):
        """
        Initializes the VinVL Feature Extractor.

        Args:
            config_file (str, optional): Path to the configuration file.
            opts (dict, optional): Additional configuration options.
            add_od_labels (bool, optional): Whether to add object detection labels to input.
        """
        super(VinVLFeatureExtractor, self).__init__(config_file, opts)
        self.add_od_labels = add_od_labels
        

    def _prepare_image(self, img):
        """
        Prepares an image for feature extraction.

        Args:
            img: Input image (file path, URL, PIL.Image, numpy array, or tensor).

        Returns:
            PIL.Image: The prepared PIL.Image in RGB format.
        """
        if isinstance(img, str):
            # File path or URL
            if img.startswith("http://") or img.startswith("https://"):
                response = requests.get(img)
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            # PIL.Image
            img = img.convert("RGB")
        elif isinstance(img, np.ndarray):
            # NumPy array (assume RGB)
            img = Image.fromarray(img)
        elif torch.is_tensor(img):
            # Tensor (assume CHW format)
            img = img.permute(1, 2, 0).cpu().numpy()  # Convert CHW to HWC
            img = Image.fromarray(np.uint8(img))
        else:
            raise ValueError("Unsupported image input type.")
        return img

    def __call__(self, imgs):
        """
        Extract features from images.

        Args:
            imgs: Single image or a batch of images (list or single instance).

        Returns:
            List[dict]: List of extracted features for each image.
        """
        if not isinstance(imgs, list):
            imgs = [imgs]  # Convert to batch format

        results = []
        for img in imgs:
            # Prepare the image
            img = self._prepare_image(img)

            # Apply transforms
            img_tensor, _ = self.transforms(img, target=None)
            img_tensor = img_tensor.to(self.device)

            # Perform inference
            with torch.no_grad():
                prediction = self.model(img_tensor)
                prediction = prediction[0].to("cpu")

            # Scale predictions to original image size
            img_width, img_height = img.size
            prediction = prediction.resize((img_width, img_height))

            # Extract features
            boxes = prediction.bbox.tolist()
            classes = [self.idx2label[c] for c in prediction.get_field("labels").tolist()]
            scores = prediction.get_field("scores").tolist()
            features = prediction.get_field("box_features").cpu().numpy()
            spatial_features = encode_spatial_features(features, (img_width, img_height), mode="xyxy")

            
            img_feats = torch.tensor(np.concatenate((features, spatial_features),axis=1),
                                    dtype=torch.float32).reshape((len(boxes),-1))
           

            # Prepare object detection labels if enabled
            if self.add_od_labels:
                od_labels = " ".join(classes)
            else:
                od_labels = None
            
           
            results.append({
                "boxes": boxes,
                "classes": classes,
                "scores": scores,
                "img_feats": img_feats,
                "od_labels":od_labels,
                "spatial_features": spatial_features,
            })

        return results

