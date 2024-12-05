from abc import ABC, abstractmethod
from typing import Any, Union, List


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.
    """

    @abstractmethod
    def extract_features(self, inputs: Union[str, List[Union[str, Any]]]) -> List[Any]:
        """
        Extract features from the given input(s).

        Args:
            inputs (Union[str, List[str], List[Any]]): Paths, URLs, or other representations of input images.

        Returns:
            List[Any]: Extracted features for each input.
        """
        pass

    def __call__(self, inputs: Union[str, List[Union[str, Any]]]) -> List[Any]:
        """
        Make the extractor callable to process inputs directly.

        Args:
            inputs (Union[str, List[str], List[Any]]): Paths, URLs, or other representations of input images.

        Returns:
            List[Any]: Extracted features for each input.
        """
        return self.extract_features(inputs)
