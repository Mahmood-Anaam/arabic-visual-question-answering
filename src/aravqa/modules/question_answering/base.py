from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List

class QuestionAnswerer(ABC):
    """Abstract base class for question answering models, handling various question inputs and generating multiple answers per question."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the QuestionAnswerer with the given configuration.

        Args:
            config: A dictionary containing configuration parameters for the model.
        """
        self.config = config
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """
        Loads the pre-trained question answering model.  This method must be implemented by concrete subclasses.
        """
        pass

    @abstractmethod
    def answer_question(self, question: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Generates an answer for the given question or list of questions.

        Args:
            question: A single question string or a list of question strings.

        Returns:
            A single answer string or a list of answer strings corresponding to the input questions.
        """
        pass