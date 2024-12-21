import torch
from llms.configuration.prompts import PROMPTS
from google.colab import userdata

class Config:

  VDS_PATH = "MahmoodAnaam/OKVQA-Encoder-Violet-Captions"
  BDS_PATH = "MahmoodAnaam/OKVQA-VinVL-BiT-Captions"
  SPLIT = "validation"
  LANGUAGE = "ar"
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  BATCH_SIZE = 5
  CAPTIONS = ["bit","violet"]
  PROMPT_TEMPLATE = PROMPTS["default"]
  SYSTEM_INSTRUCTION = None

  API_KEY = userdata.get('GEMINI_API_KEY')
  MODEL_NAME = "models/gemini-1.5-flash"
  GENERATION_CONFIG = {  
        "temperature": 1.0,  
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 20,  
        "response_mime_type": "text/plain",  
    }


  