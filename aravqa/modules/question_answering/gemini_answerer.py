import google.generativeai as genai
from typing import List, Union
from .base import BaseQuestionAnswerer
import warnings
warnings.filterwarnings("ignore")

class GeminiAnswerer(BaseQuestionAnswerer):
    """
    Implementation of the Gemini question-answering model.
    """

    def __init__(self, config):
        """
        Initializes the GeminiAnswerer with the provided configuration.

        Args:
            config: Configuration object containing model-specific parameters.
        """
        super().__init__(config)
        self.load_model()

    def load_model(self):
        """
        Loads the Gemini model and configures the API key.
        """
        genai.configure(api_key=self.config.API_KEY)
        self.model = genai.GenerativeModel(
            self.config.MODEL_NAME,
            system_instruction=self.config.SYSTEM_INSTRUCTION,
            generation_config=self.config.GENERATION_CONFIG,
        )
    

    def generate(self, prompts: Union[str, List[str]]) -> List[str]:
        """
        Generates answers based on the given prompts.

        Args:
            prompts: A single prompt or a list of prompts.

        Returns:
            List[str]: A list of generated answers.

        Raises:
            Exception: If there's an error during the API call or response parsing.
        """
        answers = []
        # Ensure prompts are in a list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        for index, prompt in enumerate(prompts):
            try:
                response = self.model.generate_content(prompt)
                answers.append(response.text.strip())
            except Exception as e:
                answers.append(f"Error")
                print(f"Error generating answer for prompt {index}: {e}")
        
        return answers
    
    def __str__(self):
        """
        Returns the string representation of the model.
        """
        return self.model.__str__()
    

    def __repr__(self):
        """
        Returns the string representation of the model.
        """
        return self.__str__()
