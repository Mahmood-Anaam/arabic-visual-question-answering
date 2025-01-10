from transformers import pipeline
from arabert.preprocess import ArabertPreprocessor
from typing import List, Union
from .base import BaseQuestionAnswerer
import re
import warnings
warnings.filterwarnings("ignore")

class AraGPT2Answerer(BaseQuestionAnswerer):
    """
    Implementation of the AraGPT2 question-answering model.
    """

    def __init__(self, config):
        """
        Initializes the AraGPT2Answerer with the provided configuration.

        Args:
            config: Configuration object containing model-specific parameters.
        """
        super().__init__(config)
        self.load_model()

    def load_model(self):
        """
        Loads the AraGPT2 model and tokenizer.
        """
        self.arabert_processor = ArabertPreprocessor(model_name=self.config.MODEL_NAME.split("/")[-1])
        self.pipeline = pipeline("text-generation",
                                 model=self.config.MODEL_NAME,
                                 trust_remote_code=True,
                                 device=self.config.DEVICE,
                                 
                                 )
        

    def generate(self, prompts: Union[str, List[str]]) -> List[str]:
        """
        Generates answers based on the given prompts.

        Args:
            prompts: A single prompt or a list of prompts.

        Returns:
            List[str]: A list of generated answers.
        """
        answers = []
        # Ensure prompts are in a list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        for index, prompt in enumerate(prompts):
            try:
                prompt_prep = self.arabert_processor.preprocess(prompt)
                gen_answer =  self.pipeline(
                    prompt_prep,
                    pad_token_id=0,
                    num_return_sequences = 1,
                    **self.config.GENERATION_CONFIG,
                )[0]['generated_text']

                gen_answer = gen_answer[len(prompt_prep):]
                gen_answer = re.sub("[^\u0621-\u063A\u0641-\u064A\u0660-\u0669\u0671-\u0673a-zA-Z ]","",gen_answer)
                answers.append(gen_answer.strip())
            except Exception as e:
                answers.append(f"Error")
                print(f"Error generating answer for prompt {index}: {e}")
        
        return answers
    
    def __str__(self):
        """
        Returns the string representation of the model.
        """
        return self.pipeline.model.__str__()
    

    def __repr__(self):
        """
        Returns the string representation of the model.
        """
        return self.model.__str__()


