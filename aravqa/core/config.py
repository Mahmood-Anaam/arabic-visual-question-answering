import torch
import textwrap


class CaptionSelection:
    """
    Enumeration-like class for caption selection strategies.
    """
    NONE = 'none'
    RANDOM = "random"  # Randomly select captions
    HIGH_SIMILARITY_QUESTION = "high_similarity_question"  # Select captions with high similarity to the question
    LOW_SIMILARITY_QUESTION = "low_similarity_question"  # Select captions with low similarity to the question
    HIGH_SIMILARITY_ANSWERS = "high_similarity_answers"  # Select captions with high similarity to the answers
    LOW_SIMILARITY_ANSWERS = "low_similarity_answers"  # Select captions with low similarity to the answers
    HIGH_SIMILARITY_QUESTION_ANSWERS = "high_similarity_question_answers"  # Select captions with high similarity to both question and answers
    LOW_SIMILARITY_QUESTION_ANSWERS = "low_similarity_question_answers"  # Select captions with low similarity to both question and answers


class Config:
    """
    Configuration class for managing global settings in the system.
    """

    # -------------------- Dataset Paths --------------------
    VDS_PATH = "MahmoodAnaam/OKVQA-Encoder-Violet-Captions"  # Path for Violet captions
    BDS_PATH = "MahmoodAnaam/OKVQA-VinVL-BiT-Captions"  # Path for BiT captions
    SPLIT = "validation"  # Dataset split (e.g., validation, test)
    USERNAME = "MahmoodAnaam"  # Hugging Face username for loading datasets

    # -------------------- Language and Device --------------------
    LANGUAGE = "ar"  # Target language for questions and answers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically detect the device (CPU/GPU)

    # -------------------- Batch and Caption Settings --------------------
    BATCH_SIZE = 20  # Batch size for processing
    

    # Number of captions to select (-1 means all captions)
    NUM_CAPTIONS = -1  # If positive, limits the number of captions per image
    CAPTION_SELECTION = CaptionSelection.NONE  # Default caption selection strategy
    RANDOM_SEED = 42  # Random seed for reproducibility
    CAPTIONS = ["bit", "violet"]  # Specify which captioning models to include (e.g., BiT and Violet,None)
    CAPTIONS_SEPARATOR = "\n"  # Default separator between captions (e.g., new line)
    PATH_RESULT_FILE = f"./{CAPTION_SELECTION}-{'-'.join(CAPTIONS)}.csv"

    # -------------------- Prompt and Instruction Settings --------------------

    # Optional System Instruction: General guideline for the LLM. [str,None]
    SYSTEM_INSTRUCTION = None
    
    # Prompt Template: Structured format for generating questions and answers
    PROMPT_TEMPLATE = textwrap.dedent(
        """
        Analyze the following image captions and answer the given question in the same language:
        Captions:{context}
        Question:{question}
        Answer concisely:
        """
    ).strip()


    # -------------------- API and Model Configuration --------------------
    API_KEY = None  # API key for external services (e.g., Google Generative AI)
    MODEL_NAME = "models/gemini-1.5-flash"  # Model name for text generation

    # -------------------- Text Generation Settings --------------------
    GENERATION_CONFIG = {
        "temperature": 0.0,  # Determines randomness in the output
        "top_p": 0.95,  # Nucleus sampling probability threshold
        "top_k": 40,  # Limits sampling to the top-k tokens
        "max_output_tokens": 20,  # Maximum length of generated responses
        "response_mime_type": "text/plain",  # Format of the output
    }
