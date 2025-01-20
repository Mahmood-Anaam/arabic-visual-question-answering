from typing import List, Dict, Callable, Optional
from aravqa.modules.evaluation import BLEUEvaluator

def compute_bleu_score(predictions: List[str], references: List[List[str]], max_order: int = 2) -> float:
    """
    Computes the BLEU score for a set of predictions and references.

    Args:
        predictions (List[str]): List of predicted sentences.
        references (List[List[str]]): List of lists of reference sentences.
        max_order (int): The maximum n-gram order to compute BLEU score.

    Returns:
        float: The computed BLEU score. Returns float(-inf) if computation fails.

    Raises:
        ValueError: If predictions or references are empty, or if their lengths do not match.
    """
    evaluator = BLEUEvaluator(max_order=max_order)
    return evaluator._compute_bleu_score(predictions, references)['bleu']
  


def prepare_answers(answers: Dict, language: str) -> List[Dict]:
    """
    Prepares answers by structuring them into a consistent format.

    Args:
        answers (Dict): A dictionary containing answers and their metadata.
        language (str): The target language for answers.

    Returns:
        List[Dict]: A list of structured answer dictionaries.
    """
   
   
    if "raw" in answers.keys():
        return [
            {
                "id": answers["id"][index],
                "answer": answer,
                "confidence": answers["confidence"][index],
                 "raw": answers.get(f"raw_{language}", [])[index] 
            }
            for index, answer in enumerate(answers.get(language, []))
        ]
    else:
        return [
            {
                "id": answers["id"][index],
                "answer": answer,
                "confidence": answers["confidence"][index],
            }
            for index, answer in enumerate(answers.get(language, []))
        ]
        


def prepare_captions(captions: List[Dict]) -> List[Dict]:
    """
    Prepares captions by structuring them into a consistent format.

    Args:
        captions (List[Dict]): A list of caption dictionaries.

    Returns:
        List[Dict]: A list of structured caption dictionaries.
    """
    return [
        {
            "id": index,
            "caption": cap.get("caption", None)
        }
        for index, cap in enumerate(captions)
    ]


def process_dataset_example(example: Dict, language: str) -> Dict:
    """
    Processes a single example from the dataset.

    Args:
        example (Dict): A single dataset example.
        language (str): The target language for questions and answers.

    Returns:
        Dict: The processed example with formatted questions, answers, and captions.
    """
    question = example["question"].get(language)
    answers = prepare_answers(example.get("answers", {}), language)
    captions = prepare_captions(example.get("captions", []))

    if "multiple_choice_answer" in example.keys():
        return {
            "question": question,
            "answers": answers,
            "captions": captions,
            "multiple_choice_answer":example["multiple_choice_answer"].get(language,""),
            "metadata": example.get("metadata"),
            "image": example.get("image")
        }
    else:
        return {
        "question": question,
        "answers": answers,
        "captions": captions,
        "metadata": example.get("metadata"),
        "image": example.get("image")
        }


def prepare_dataset(dataset, language: str = "ar"):
    """
    Prepares a dataset by transforming its structure.

    Args:
        dataset: The dataset to process.
        language (str): The target language for questions and answers (default: "ar").

    Returns:
        The transformed dataset with uniform formatting.
    """
    return dataset.map(lambda example: process_dataset_example(example, language=language))


def compute_similarity_question_captions(
    question: str, captions: List[Dict], scorer: Callable
) -> List[Dict]:
    """
    Computes the similarity between a question and a list of captions using a scoring function.

    Args:
        question (str): The question text.
        captions (List[Dict]): A list of caption dictionaries.
        scorer (Callable): A similarity scoring function.

    Returns:
        List[Dict]: A list of captions with appended similarity scores.
    """
    return [
        {
            **cap,
            "similarity_question": scorer([cap.get("caption",None)],[[question]])
        }
        for cap in captions
    ]


def compute_similarity_answers_captions(
    answers: List[Dict], captions: List[Dict], scorer: Callable
) -> List[Dict]:
    """
    Computes the similarity between a list of answers and captions using a scoring function.

    Args:
        answers (List[Dict]): A list of answer dictionaries.
        captions (List[Dict]): A list of caption dictionaries.
        scorer (Callable): A similarity scoring function.

    Returns:
        List[Dict]: A list of captions with appended similarity scores to answers.
    """
    answer_texts = [answer.get("answer","") for answer in answers]
    return [
        {
            **cap,
            "similarity_answers": scorer([cap.get("caption", None)], [answer_texts])
        }
        for cap in captions
    ]


def compute_similarity_captions(
    dataset,
    question_similarity_scorer: Optional[Callable] = None,
    answer_similarity_scorer: Optional[Callable] = None,
):
    """
    Computes the similarity between questions, answers, and captions in a dataset.

    Args:
        dataset: The dataset to process.
        question_similarity_scorer (Optional[Callable]): A scoring function for question-caption similarity.
        answer_similarity_scorer (Optional[Callable]): A scoring function for answer-caption similarity.

    Returns:
        The processed dataset with added similarity scores.
    """
    def process_example(example):
        if question_similarity_scorer:
            example["captions"] = compute_similarity_question_captions(
                example["question"], example["captions"], question_similarity_scorer
            )
        if answer_similarity_scorer:
            example["captions"] = compute_similarity_answers_captions(
                example["answers"], example["captions"], answer_similarity_scorer
            )
        return example

    return dataset.map(process_example)


