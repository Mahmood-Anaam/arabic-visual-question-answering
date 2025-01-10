from abc import ABC, abstractmethod
import json
from typing import List, Dict

class BaseEvaluator(ABC):
    """
    Abstract base class for evaluation modules.
    """
    def __init__(self,**kwargs):
        """
        Initializes the evaluator with optional arguments.
        """
        self.kwargs = kwargs


    @abstractmethod
    def evaluate(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score.
        """
        pass

    def  export(self,results:Dict,path:str)->None:
        """
        Exports the evaluation results to a JSON file.

        Args:
            results (Dict): The evaluation results.
            path (str): The path to the output file.
        """
        with open(path, "w") as file:
            json.dump(results, file, indent=4)

    def __call__(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates predictions against references.

        Args:
            predictions (List[str]): List of predicted sentences.
            references (List[List[str]]): List of lists of reference sentences.

        Returns:
            Dict: The computed evaluation score.
        """
        return self.evaluate(predictions, references)
    

