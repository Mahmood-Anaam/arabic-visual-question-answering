from abc import ABC, abstractmethod
from typing import List, Union,Dict
from tqdm import tqdm

class BaseQuestionAnswerer(ABC):
    """
    Abstract Base Class for question-answering models.
    Provides a unified interface for question-answering models.
    """

    def __init__(self, config):
        """
        Initializes the question-answering model with the provided configuration.

        Args:
            config: Configuration object containing model-specific parameters.
        """
        self.config = config

    @abstractmethod
    def load_model(self):
        """
        Loads the question-answering model and its dependencies.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def generate(
        self, 
        prompts: Union[str, List[str]]
    ) -> List[str]:
        """
        Generates answers based on the given prompt(s).

        Args:
            prompts: A single formatted prompt or a list of prompts.

        Returns:
            List[str]: A list of generated answers.
        """
        pass

   



    def generate_from_dataloader(self, dataloader) -> Dict:
        """
        Processes the dataloader to generate predictions for each batch.

        Args:
            dataloader: The DataLoader object containing batches of data.

        Returns:
            Dict: A dictionary containing question IDs, image IDs, predictions, and ground-truth answers.
                Format:
                {
                    "question_id": List[str],
                    "image_id": List[str],
                    "answers": List[List[str]],
                    "predictions": List[str]
                }
        """
        results = {
            "question_id": [],
            "image_id": [],
            "answers": [],
            "predictions": []
        }

        for batch in tqdm(dataloader, desc="Generating predictions from dataloader"):
            # Generate predictions using prompts from the batch
            predictions = self.generate(batch["prompts"])

            # Append batch data to results
            results["question_id"].extend(batch["question_id"])
            results["image_id"].extend(batch["image_id"])
            results["answers"].extend(batch["answers"])
            results["predictions"].extend(predictions)

        return results

    
    def __call__(self, prompts: Union[str, List[str]]) -> List[str]:
        """
        Generates answers based on the given prompt(s).

        Args:
            prompts: A single formatted prompt or a list of prompts.

        Returns:
            List[str]: A list of generated answers.
        """
        return self.generate(prompts)
    

