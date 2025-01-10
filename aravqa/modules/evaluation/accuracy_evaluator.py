import re
from typing import List, Dict
from .base import BaseEvaluator
from tqdm import tqdm

class AccuracyEvaluator(BaseEvaluator):
    """
    Evaluates the accuracy of machine-generated answers against human-provided answers
    based on the VQA accuracy formula, adapted for Arabic.
    """

    def __init__(self):
        """
        Initializes the AccuracyEvaluator with predefined mappings and patterns for Arabic processing.
        """
        super().__init__()
        # Mapping for numbers and common contractions in Arabic
        self.manual_map = {
            'صفر': '0', 'واحد': '1', 'اثنان': '2', 'ثلاثة': '3',
            'أربعة': '4', 'خمسة': '5', 'ستة': '6', 'سبعة': '7',
            'ثمانية': '8', 'تسعة': '9', 'عشرة': '10'
        }
        self.articles = ['ال', 'و', 'ف', 'ب']  # Common Arabic prefixes/articles to remove
        self.punctuation = ['،', '.', '؟', '!', '؛', ':', '-', '_', '"', "'", '(', ')', '[', ']']
        self.strip_punctuation = re.compile(r"[{}]".format("".join(self.punctuation)))

    def evaluate(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Evaluates the accuracy of predictions against references.

        Args:
            predictions (List[str]): List of predicted answers.
            references (List[List[str]]): List of lists containing human-provided answers.

        Returns:
            Dict: A dictionary containing the overall accuracy and individual question accuracies.
        """
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty.")
        if len(predictions) != len(references):
            raise ValueError("The number of predictions must match the number of reference sets.")

 
        individual_accuracies = []

        for pred, refs in tqdm(zip(predictions, references), total=len(predictions), desc="Evaluating Accuracy scores"):
            processed_pred = self._process_text(pred)
            processed_refs = [self._process_text(ref) for ref in refs]

            # Compute accuracy for this prediction
            acc = self._compute_accuracy(processed_pred, processed_refs)
            individual_accuracies.append(acc)

        # Return overall accuracy and individual accuracies
        return {
            "overall_accuracy": sum(individual_accuracies) / len(individual_accuracies),
            "accuracy": individual_accuracies
        }

    def _compute_accuracy(self, prediction: str, references: List[str]) -> float:
        """
        Computes accuracy for a single prediction against multiple references.

        Args:
            prediction (str): The machine-generated answer.
            references (List[str]): The list of human-provided reference answers.

        Returns:
            float: The computed accuracy score (0 to 1).
        """
        num_matching = references.count(prediction)
        return min(num_matching / 3.0, 1.0)

    def _process_text(self, text: str) -> str:
        """
        Normalizes and processes Arabic text by removing unnecessary characters and formatting.

        Args:
            text (str): The input text.

        Returns:
            str: The processed text.
        """
        # Lowercase the text
        text = text.lower()

        # Remove punctuation
        text = self.strip_punctuation.sub("", text)

        # Remove articles and prefixes
        words = text.split()
        words = [word.lstrip("".join(self.articles)) for word in words]

        # Convert number words to digits
        words = [self.manual_map.get(word, word) for word in words]

        return " ".join(words)

    def export(self, results: Dict, path: str) -> None:
        """
        Exports the evaluation results to a JSON file.

        Args:
            results (Dict): The evaluation results.
            path (str): Path to the output file.
        """
        super().export(results, path)
