import google.generativeai as genai
from typing import List, Dict, Any, Optional
from llms.configuration import Config

class GeminiLLM:
    """
    A general-purpose class for interacting with Large Language Models (LLMs),
    specifically designed for Google's Generative AI.
    """

    def __init__(self, config=Config):
        """
        Initializes the LLM with the provided configuration.

        Args:
            config: An instance of the Config class containing API key, model name, and other settings.
        """
        self.config = config
        genai.configure(api_key=self.config.API_KEY)
        self.model = genai.GenerativeModel(self.config.MODEL_NAME,
                                           system_instruction=self.config.SYSTEM_INSTRUCTION,
                                           generation_config=self.config.GENERATION_CONFIG,
                                           )




    def generate(self, prompts: List[str]):
        """
        Generates responses for a batch of prompts.

        Args:
            prompts: A list of prompt strings.
            
        Returns:
             A list of dictionaries, where each dictionary contains the generation results 
             for a single prompt (e.g., 'text', 'safetyAttributes').  This is following
             the way the underlying `palm.generate_text` behaves.
        """
        results = []
        for prompt in prompts:
            full_prompt = prompt
            response = self.model.generate_content(full_prompt)
            results.append(response.text) 
        return results

    def generate_single(self, prompt: str):
        """
        Generates a response for a single prompt.

        Args:
            prompt: The prompt string.
        Returns:
            A dictionary containing the generation results or None if the generation fails.
        """
        full_prompt = prompt
        response = self.model.generate_content(full_prompt)

        return response.text



